//! `gameinput.dll`: input devices from the host. Gamepads come from XInput
//! (host slots 0-3); while none is connected, the keyboard drives one
//! stand-in gamepad (`FALLBACK_BUTTONS`). The keyboard is a device of its own
//! too. A poll thread turns host state into readings (a new one per change)
//! and runs device callbacks on connect and disconnect.
#![allow(non_snake_case)]
// Title code calls these through the objects, passing pointers the real API
// would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, LazyLock, Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use super::lock;
use super::xasync::{E_INVALIDARG, E_NOINTERFACE, E_NOTIMPL};
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleExport, HleModule};

/// The documented errors.
pub const DEVICE_DISCONNECTED: HRESULT = 0x838A_0001_u32 as i32;
pub const DEVICE_NOT_FOUND: HRESULT = 0x838A_0002_u32 as i32;
pub const READING_NOT_FOUND: HRESULT = 0x838A_0003_u32 as i32;
pub const REFERENCE_TOO_OLD: HRESULT = 0x838A_0004_u32 as i32;

/// Input kinds (bits) our devices have.
const KEYBOARD_KIND: u32 = 0x10;
const GAMEPAD_KIND: u32 = 0x4_0000;
/// Device status of a connected device: connected, input enabled.
const STATUS_CONNECTED: u32 = 0x3;
const ASYNC_ENUMERATION: u32 = 1;

/// Gamepad buttons; the title maps each bit to one of its pad buttons.
const MENU: u32 = 0x1;
const VIEW: u32 = 0x2;
const A: u32 = 0x4;
const B: u32 = 0x8;
const X: u32 = 0x10;
const Y: u32 = 0x20;
const UP: u32 = 0x40;
const DOWN: u32 = 0x80;
const LEFT: u32 = 0x100;
const RIGHT: u32 = 0x200;
const LEFT_SHOULDER: u32 = 0x400;
const RIGHT_SHOULDER: u32 = 0x800;
const LEFT_STICK: u32 = 0x1000;
const RIGHT_STICK: u32 = 0x2000;

/// Devices: host pad slots, the keyboard's stand-in pad, the keyboard.
const PADS: usize = 4;
const FALLBACK: usize = PADS;
const KEYBOARD: usize = PADS + 1;
const DEVICES: usize = PADS + 2;

/// Readings kept per device for next/previous lookups.
const HISTORY: usize = 32;
const POLL: Duration = Duration::from_millis(8);

/// The keyboard's stand-in gamepad (virtual key, button); README lists it.
/// Sticks: W/A/S/D left, I/J/K/L right; triggers Z and C.
const FALLBACK_BUTTONS: [(u8, u32); 14] = [
    (0x1B, MENU), // Esc
    (0x09, VIEW), // Tab
    (0x0D, A),    // Enter
    (0x08, B),    // Backspace
    (b'X', X),
    (b'Y', Y),
    (0x26, UP), // arrows
    (0x28, DOWN),
    (0x25, LEFT),
    (0x27, RIGHT),
    (b'Q', LEFT_SHOULDER),
    (b'E', RIGHT_SHOULDER),
    (b'F', LEFT_STICK),
    (b'H', RIGHT_STICK),
];

/// PC-style bindings on the stand-in pad (`VICTORY_KBM_GTA=1`), adapted
/// from Belobug/VIctory-Belobug-AMD #295. I/J/K/L still move the camera.
#[cfg(any(windows, test))]
const GTA_BUTTONS: &[(u8, u32)] = &[
    (0x1B, MENU), // Esc: pause
    (b'P', MENU),
    (b'V', VIEW), // camera perspective
    (0xA0, A),    // left Shift: sprint
    (0xA1, A),    // right Shift
    (0x0D, A),    // Enter: confirm
    (0x20, X),    // Space: jump
    (b'X', X),
    (b'R', B),              // reload
    (0x08, B),              // Backspace: cancel
    (b'F', Y),              // enter or exit vehicle
    (0x09, LEFT_SHOULDER),  // Tab: weapon wheel
    (b'Q', RIGHT_SHOULDER), // cover / handbrake
    (0xA2, LEFT_STICK),     // left Ctrl: crouch / horn
    (0xA3, LEFT_STICK),     // right Ctrl
    (b'C', RIGHT_STICK),    // look behind
    (0x14, RIGHT_STICK),    // Caps Lock
    (b'H', RIGHT_STICK),
    (b'E', RIGHT), // interact
    (0x27, RIGHT),
    (0x26, UP),
    (b'M', UP),
    (0x28, DOWN),
    (b'Z', DOWN),
    (0x25, LEFT),
    (b'G', LEFT),
];

/// A gamepad reading: triggers 0 to 1, sticks -1 to 1 (up positive).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PadState {
    buttons: u32,
    left_trigger: f32,
    right_trigger: f32,
    left_x: f32,
    left_y: f32,
    right_x: f32,
    right_y: f32,
}

/// One key held down: scan code (0xE0xx when extended), character, key.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct KeyState {
    scan_code: u32,
    code_point: u32,
    virtual_key: u8,
    dead_key: bool,
}

const _: () = assert!(size_of::<KeyState>() == 12 && size_of::<PadState>() == 28);

/// Host input, or a fake in tests.
trait Source: Send {
    /// The pad in host slot `slot`, `None` when the slot is empty.
    fn pad(&mut self, slot: usize) -> Option<PadState>;
    /// Keys held down; none while the title's window lacks focus.
    fn keys(&mut self) -> Vec<KeyState>;
    /// Builds the stand-in pad and releases any mouse capture if a pad takes over.
    fn fallback_pad(&mut self, keys: &[KeyState], _active: bool) -> PadState {
        fallback(keys)
    }
    /// Scripted diagnostic keys (`VICTORY_KEYS`), independent of focus.
    #[cfg(any(test, feature = "diagnostics"))]
    fn scripted_keys(&mut self) -> Vec<KeyState> {
        Vec::new()
    }
    /// Motor speeds: low frequency (left), high frequency (right).
    fn rumble(&mut self, slot: usize, low: u16, high: u16);
}

fn fallback(keys: &[KeyState]) -> PadState {
    keyboard_pad(keys, &FALLBACK_BUTTONS, Some((b'Z', b'C')))
}

fn keyboard_pad(keys: &[KeyState], buttons: &[(u8, u32)], triggers: Option<(u8, u8)>) -> PadState {
    let on = |vk: u8| {
        if keys.iter().any(|k| k.virtual_key == vk) {
            1.0
        } else {
            0.0
        }
    };
    PadState {
        buttons: buttons
            .iter()
            .filter(|(vk, _)| on(*vk) != 0.0)
            .fold(0, |all, (_, bit)| all | bit),
        left_trigger: triggers.map_or(0.0, |(left, _)| on(left)),
        right_trigger: triggers.map_or(0.0, |(_, right)| on(right)),
        left_x: on(b'D') - on(b'A'),
        left_y: on(b'W') - on(b'S'),
        right_x: on(b'L') - on(b'J'),
        right_y: on(b'I') - on(b'K'),
    }
}

#[cfg(any(windows, test))]
#[derive(Default)]
struct MouseLook {
    velocity: [f32; 2],
}

#[cfg(any(windows, test))]
impl MouseLook {
    fn step(&mut self, delta: [i32; 2], elapsed: Duration, sensitivity: f32, invert_y: bool) {
        // Keep motion available between the 8 ms polls and slow title frames.
        let decay = 0.5_f32.powf(elapsed.as_secs_f32() / 0.045);
        let y_sign = if invert_y { 1.0 } else { -1.0 };
        for ((velocity, delta), sign) in self.velocity.iter_mut().zip(delta).zip([1.0, y_sign]) {
            *velocity =
                (*velocity * decay + delta as f32 * (0.022 * sensitivity) * sign).clamp(-1.0, 1.0);
            if delta == 0 && velocity.abs() < 0.015 {
                *velocity = 0.0;
            }
        }
    }

    fn pad(&self, [left, right, middle]: [bool; 3]) -> PadState {
        PadState {
            buttons: if middle { RIGHT_STICK } else { 0 },
            left_trigger: u8::from(right).into(),
            right_trigger: u8::from(left).into(),
            right_x: self.velocity[0],
            right_y: self.velocity[1],
            ..PadState::default()
        }
    }
}

/// Combines keyboard, mouse or diagnostic inputs on a single pad.
#[cfg(any(windows, test, feature = "diagnostics"))]
fn merge_pad(base: PadState, extra: PadState) -> PadState {
    PadState {
        buttons: base.buttons | extra.buttons,
        left_trigger: base.left_trigger.max(extra.left_trigger),
        right_trigger: base.right_trigger.max(extra.right_trigger),
        left_x: (base.left_x + extra.left_x).clamp(-1.0, 1.0),
        left_y: (base.left_y + extra.left_y).clamp(-1.0, 1.0),
        right_x: (base.right_x + extra.right_x).clamp(-1.0, 1.0),
        right_y: (base.right_y + extra.right_y).clamp(-1.0, 1.0),
    }
}

/// Applies `scripted` keys to the first connected host pad, so `VICTORY_KEYS`
/// works even when an XInput or virtual controller disables `FALLBACK`.
#[cfg(any(test, feature = "diagnostics"))]
fn merge_scripted(pads: &mut [Option<PadState>; PADS], scripted: &[KeyState]) {
    if scripted.is_empty() {
        return;
    }
    let extra = fallback(scripted);
    if let Some(pad) = pads.iter_mut().flatten().next() {
        *pad = merge_pad(*pad, extra);
    }
}

#[repr(C)]
struct Table<const N: usize>([*const (); N]);
// SAFETY: function addresses only.
unsafe impl<const N: usize> Sync for Table<N> {}

/// The input object; one per hub.
#[repr(C)]
pub struct Object {
    vtable: &'static Table<22>,
    hub: *const Hub,
}

#[repr(C)]
pub struct Device {
    vtable: &'static Table<21>,
    hub: *const Hub,
    index: usize,
    info: DeviceInfo,
}

/// What a device says about itself. The title reads `id` (+0x20); the rest
/// follows the public layout and is zero where we have nothing to say.
#[repr(C)]
#[derive(Default)]
#[allow(dead_code)] // read by title code
struct DeviceInfo {
    size: u32,
    vendor: u16,
    product: u16,
    revision: u16,
    interface: u8,
    collection: u8,
    usage: [u16; 2],
    versions: [u16; 8],
    id: [u8; 32],
    root_id: [u8; 32],
    family: i32,
    capabilities: u32,
    supported_input: u32,
    rumble_motors: u32,
    /// Counts and optional detail pointers.
    rest: [u64; 26],
}

const _: () = assert!(std::mem::offset_of!(DeviceInfo, id) == 0x20);

fn kind(index: usize) -> u32 {
    if index == KEYBOARD {
        KEYBOARD_KIND
    } else {
        GAMEPAD_KIND
    }
}

impl Device {
    fn new(index: usize) -> Device {
        let mut id = [0; 32];
        id[..7].copy_from_slice(b"victory");
        id[31] = index as u8 + 1;
        let info = DeviceInfo {
            size: size_of::<DeviceInfo>() as u32,
            id,
            root_id: id,
            supported_input: kind(index),
            // Low and high frequency.
            rumble_motors: if index < PADS { 0x3 } else { 0 },
            ..Default::default()
        };
        Device {
            vtable: &DEVICE,
            hub: std::ptr::null(),
            index,
            info,
        }
    }

    fn hub(&self) -> &'static Hub {
        // SAFETY: set by `Hub::new`; hubs are never freed.
        unsafe { &*self.hub }
    }
}

#[derive(PartialEq)]
enum Value {
    Pad(PadState),
    Keys(Vec<KeyState>),
}

#[repr(C)]
pub struct Reading {
    vtable: &'static Table<25>,
    refs: AtomicU32,
    device: *const Device,
    sequence: u64,
    timestamp: u64,
    value: Value,
}

impl Reading {
    fn kind(&self) -> u32 {
        match self.value {
            Value::Pad(_) => GAMEPAD_KIND,
            Value::Keys(_) => KEYBOARD_KIND,
        }
    }
}

/// A counted reference to a reading.
struct Ref(NonNull<Reading>);

impl Ref {
    fn new(reading: Reading) -> Ref {
        Ref(NonNull::from(Box::leak(Box::new(reading))))
    }

    /// Another reference, for the title.
    fn share(&self) -> *const Reading {
        reading_add_ref(self.0.as_ptr());
        self.0.as_ptr()
    }
}

impl Deref for Ref {
    type Target = Reading;
    fn deref(&self) -> &Reading {
        // SAFETY: we hold a reference, so it is alive.
        unsafe { self.0.as_ref() }
    }
}

impl Drop for Ref {
    fn drop(&mut self) {
        reading_release(self.0.as_ptr());
    }
}

type DeviceCallback = extern "system" fn(u64, *mut c_void, *const Device, u64, u32, u32);

#[derive(Clone, Copy)]
struct Registration {
    token: u64,
    /// One device's address, or 0 for any.
    device: usize,
    kind: u32,
    status: u32,
    context: usize,
    callback: DeviceCallback,
}

impl Registration {
    fn matches(&self, devices: &[Device; DEVICES], index: usize) -> bool {
        kind(index) & self.kind != 0
            && (self.device == 0 || self.device == &devices[index] as *const Device as usize)
    }

    fn event(&self, device: usize, timestamp: u64, current: u32, previous: u32) -> Event {
        Event {
            registration: *self,
            device,
            timestamp,
            current,
            previous,
        }
    }
}

struct Event {
    registration: Registration,
    device: usize,
    timestamp: u64,
    current: u32,
    previous: u32,
}

#[derive(Default)]
struct Slot {
    connected: bool,
    sequence: u64,
    history: VecDeque<Ref>,
}

struct State {
    slots: [Slot; DEVICES],
    callbacks: Vec<Registration>,
    /// Asynchronous enumerations, delivered by the next poll.
    pending: Vec<Event>,
    /// Device callbacks running now: token and thread.
    running: Vec<(u64, ThreadId)>,
}

struct PollSource {
    input: Box<dyn Source>,
    /// Host slots that have reported non-rest input on their current connection.
    seen_input: [bool; PADS],
}

impl PollSource {
    fn pads(&mut self) -> [Option<PadState>; PADS] {
        std::array::from_fn(|slot| {
            // Some wireless receivers answer while their pad is switched off.
            let pad = self.input.pad(slot);
            self.seen_input[slot] =
                pad.is_some_and(|p| self.seen_input[slot] || p != PadState::default());
            pad.filter(|_| self.seen_input[slot])
        })
    }
}

pub struct Hub {
    object: Object,
    devices: [Device; DEVICES],
    /// Apart from the state: XInput can be slow, and the title reads
    /// readings every frame.
    source: Mutex<PollSource>,
    state: Mutex<State>,
    /// A device callback returned.
    returned: Condvar,
}

// SAFETY: the back pointers are set before the hub is shared and never
// change; what changes sits behind the mutex or is atomic.
unsafe impl Send for Hub {}
// SAFETY: as above.
unsafe impl Sync for Hub {}

static START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Microseconds, the clock of every timestamp we hand out.
fn now() -> u64 {
    START.elapsed().as_micros() as u64
}

impl Hub {
    fn new(source: Box<dyn Source>) -> &'static Hub {
        let hub = Box::into_raw(Box::new(Hub {
            object: Object {
                vtable: &INPUT,
                hub: std::ptr::null(),
            },
            devices: std::array::from_fn(Device::new),
            source: Mutex::new(PollSource {
                input: source,
                seen_input: [false; PADS],
            }),
            state: Mutex::new(State {
                slots: Default::default(),
                callbacks: Vec::new(),
                pending: Vec::new(),
                running: Vec::new(),
            }),
            returned: Condvar::new(),
        }));
        // SAFETY: just allocated and never freed; nothing else sees it yet.
        unsafe {
            (*hub).object.hub = hub;
            for device in &mut (*hub).devices {
                device.hub = hub;
            }
            &*hub
        }
    }

    /// A hub over the host's input, polled from now on.
    fn start(source: Box<dyn Source>) -> &'static Hub {
        let hub = Hub::new(source);
        hub.poll();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(POLL);
                hub.poll();
            }
        });
        hub
    }

    fn state(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    /// Reads the host once; device callbacks run on the calling thread.
    fn poll(&self) {
        let events = {
            let mut source = lock(&self.source);
            let pads = source.pads();
            let keys = source.input.keys();
            let active = pads.iter().all(Option::is_none);
            let fallback = source.input.fallback_pad(&keys, active);
            #[cfg(any(test, feature = "diagnostics"))]
            let scripted = source.input.scripted_keys();
            #[cfg(not(any(test, feature = "diagnostics")))]
            let scripted: [KeyState; 0] = [];
            self.state().poll(
                &self.devices,
                pads,
                keys,
                active.then_some(fallback),
                &scripted,
            )
        };
        self.deliver(events);
    }

    /// Runs callbacks, never under our lock: they call back into us. One
    /// unregistered since the event was taken does not run, and
    /// `UnregisterCallback` waits for one that runs.
    fn deliver(&self, events: Vec<Event>) {
        let me = std::thread::current().id();
        for e in events {
            let r = e.registration;
            {
                let mut state = self.state();
                if !state.callbacks.iter().any(|c| c.token == r.token) {
                    continue;
                }
                state.running.push((r.token, me));
            }
            (r.callback)(
                r.token,
                r.context as *mut c_void,
                &self.devices[e.device],
                e.timestamp,
                e.current,
                e.previous,
            );
            let mut state = self.state();
            if let Some(i) = state.running.iter().position(|&x| x == (r.token, me)) {
                state.running.swap_remove(i);
            }
            self.returned.notify_all();
        }
    }

    fn index(&self, device: *const Device) -> Option<usize> {
        self.devices.iter().position(|d| std::ptr::eq(d, device))
    }
}

impl State {
    fn poll(
        &mut self,
        devices: &[Device; DEVICES],
        mut pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
        stand_in: Option<PadState>,
        scripted: &[KeyState],
    ) -> Vec<Event> {
        let now = now();
        let mut events = std::mem::take(&mut self.pending);
        #[cfg(any(test, feature = "diagnostics"))]
        merge_scripted(&mut pads, scripted);
        #[cfg(not(any(test, feature = "diagnostics")))]
        let _ = (&mut pads, scripted);
        for (slot, pad) in pads.into_iter().enumerate() {
            self.update(devices, slot, pad.map(Value::Pad), now, &mut events);
        }
        let stand_in = stand_in.map(|base| {
            #[cfg(any(test, feature = "diagnostics"))]
            if !scripted.is_empty() {
                return Value::Pad(merge_pad(base, fallback(scripted)));
            }
            Value::Pad(base)
        });
        self.update(devices, FALLBACK, stand_in, now, &mut events);
        self.update(devices, KEYBOARD, Some(Value::Keys(keys)), now, &mut events);
        events
    }

    /// Records a device's state: a reading when it changed, callbacks when
    /// the device came or went.
    fn update(
        &mut self,
        devices: &[Device; DEVICES],
        index: usize,
        value: Option<Value>,
        now: u64,
        events: &mut Vec<Event>,
    ) {
        let slot = &mut self.slots[index];
        let was = slot.connected;
        slot.connected = value.is_some();
        if let Some(value) = value
            && (!was || slot.history.back().is_none_or(|r| r.value != value))
        {
            slot.sequence += 1;
            if slot.history.len() == HISTORY {
                slot.history.pop_front();
            }
            slot.history.push_back(Ref::new(Reading {
                vtable: &READING,
                refs: AtomicU32::new(1),
                device: &devices[index],
                sequence: slot.sequence,
                timestamp: now,
                value,
            }));
        }
        if was != slot.connected {
            tracing::info!(device = index, connected = !was, "input device");
            let (current, previous) = if was {
                (0, STATUS_CONNECTED)
            } else {
                (STATUS_CONNECTED, 0)
            };
            events.extend(
                self.callbacks
                    .iter()
                    .filter(|r| r.matches(devices, index) && r.status & STATUS_CONNECTED != 0)
                    .map(|r| r.event(index, now, current, previous)),
            );
        }
    }
}

#[cfg(windows)]
fn host() -> Box<dyn Source> {
    Box::new(host::Host::default())
}

#[cfg(not(windows))]
fn host() -> Box<dyn Source> {
    struct Nothing;
    impl Source for Nothing {
        fn pad(&mut self, _: usize) -> Option<PadState> {
            None
        }
        fn keys(&mut self) -> Vec<KeyState> {
            Vec::new()
        }
        fn rumble(&mut self, _: usize, _: u16, _: u16) {}
    }
    Box::new(Nothing)
}

static HUB: LazyLock<&'static Hub> = LazyLock::new(|| Hub::start(host()));

type This = *const Object;
type Out = *mut *const c_void;

/// The DLL's only export (ordinal 1): the input object, static and shared.
pub extern "system" fn GameInputCreate(out: *mut This) -> HRESULT {
    tracing::info!("GameInputCreate");
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes room for a pointer.
    unsafe { out.write(&HUB.object) };
    S_OK
}

fn hub(this: This) -> &'static Hub {
    // SAFETY: `this` is a hub's object; hubs are never freed.
    unsafe { &*(*this).hub }
}

/// Clears a COM out pointer and returns `hr`.
fn fail(out: Out, hr: HRESULT) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(std::ptr::null()) };
    }
    hr
}

/// Gives the title a reference to `reading`, or fails as with none.
fn hand(out: Out, reading: Option<&Ref>) -> HRESULT {
    match reading {
        _ if out.is_null() => E_INVALIDARG,
        Some(reading) => {
            // SAFETY: checked non-null; the title passes room for a pointer.
            unsafe { out.write(reading.share().cast()) };
            S_OK
        }
        None => fail(out, READING_NOT_FOUND),
    }
}

extern "system" fn no_interface(_: *const c_void, _iid: *const c_void, out: Out) -> HRESULT {
    tracing::warn!("input QueryInterface not implemented");
    fail(out, E_NOINTERFACE)
}
/// The input object and devices live as long as the process.
extern "system" fn add_ref(_: *const c_void) -> u32 {
    1
}

extern "system" fn timestamp(_: This) -> u64 {
    now()
}

extern "system" fn current_reading(
    this: This,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let hub = hub(this);
    let state = hub.state();
    let newest = |i: usize| {
        state.slots[i]
            .history
            .back()
            .filter(|r| r.kind() & kind != 0)
    };
    let found = if device.is_null() {
        (0..DEVICES)
            .filter(|&i| state.slots[i].connected)
            .filter_map(newest)
            .max_by_key(|r| r.timestamp)
    } else {
        match hub.index(device) {
            None => return fail(out, E_INVALIDARG),
            Some(i) if !state.slots[i].connected => return fail(out, DEVICE_DISCONNECTED),
            Some(i) => newest(i),
        }
    };
    tracing::trace!(
        kind = format_args!("{kind:#x}"),
        device = ?hub.index(device),
        sequence = found.map(|r| r.sequence),
        "current reading"
    );
    hand(out, found)
}

/// The reading after (`forward`) or before `reference`: on `device`, or with
/// none given on any device, in time order (ties by device, then sequence).
fn relative(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
    forward: bool,
) -> HRESULT {
    if reference.is_null() {
        return fail(out, E_INVALIDARG);
    }
    let hub = hub(this);
    // SAFETY: a reading we handed out and the title still holds.
    let reference = unsafe { &*reference };
    let Some(index) = hub.index(reference.device) else {
        return fail(out, E_INVALIDARG);
    };
    if !device.is_null() && !std::ptr::eq(device, reference.device) {
        return fail(out, E_INVALIDARG);
    }
    let state = hub.state();
    if state.slots[index]
        .history
        .front()
        .is_none_or(|oldest| reference.sequence < oldest.sequence)
    {
        return fail(out, REFERENCE_TOO_OLD);
    }
    // On one device this order is its sequence order.
    let at = (reference.timestamp, index, reference.sequence);
    let devices = if device.is_null() {
        0..DEVICES
    } else {
        index..index + 1
    };
    let candidates = devices
        .flat_map(|i| {
            let history = &state.slots[i].history;
            history
                .iter()
                .map(move |r| ((r.timestamp, i, r.sequence), r))
        })
        .filter(|(_, r)| r.kind() & kind != 0);
    let found = if forward {
        candidates
            .filter(|(key, _)| *key > at)
            .min_by_key(|(key, _)| *key)
    } else {
        candidates
            .filter(|(key, _)| *key < at)
            .max_by_key(|(key, _)| *key)
    };
    hand(out, found.map(|(_, r)| r))
}
extern "system" fn next_reading(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    relative(this, reference, kind, device, out, true)
}
extern "system" fn previous_reading(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    relative(this, reference, kind, device, out, false)
}
/// The device's reading at `time`, among those still kept.
extern "system" fn temporal_reading(
    this: This,
    time: u64,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let hub = hub(this);
    let Some(index) = hub.index(device) else {
        return fail(out, E_INVALIDARG);
    };
    let state = hub.state();
    let history = &state.slots[index].history;
    hand(out, history.iter().rev().find(|r| r.timestamp <= time))
}

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

fn token(out: *mut u64) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(NEXT_TOKEN.fetch_add(1, Ordering::Relaxed)) };
    }
    S_OK
}

// shortcut: reading callbacks never run; Grand Theft Auto VI registers none
// (it polls).
extern "system" fn register_reading(
    _: This,
    _device: *const c_void,
    _kind: u32,
    _threshold: f32,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    tracing::warn!("input reading callbacks are never called");
    token(out)
}

/// Device callbacks run on our poll thread when a device comes or goes.
/// Enumeration reports the devices already there: blocking on the caller's
/// thread before we return (the title registers holding a lock its callback
/// takes again), asynchronous on the next poll.
extern "system" fn register_device(
    this: This,
    device: *const Device,
    kind: u32,
    status: u32,
    enumeration: u32,
    context: *mut c_void,
    callback: Option<DeviceCallback>,
    out: *mut u64,
) -> HRESULT {
    let Some(callback) = callback else {
        return E_INVALIDARG;
    };
    let hub = hub(this);
    let registration = Registration {
        token: NEXT_TOKEN.fetch_add(1, Ordering::Relaxed),
        device: device as usize,
        kind,
        status,
        context: context as usize,
        callback,
    };
    let mut found = Vec::new();
    {
        let mut state = hub.state();
        if enumeration != 0 && status & STATUS_CONNECTED != 0 {
            let now = now();
            found = (0..DEVICES)
                .filter(|&i| state.slots[i].connected && registration.matches(&hub.devices, i))
                .map(|i| registration.event(i, now, STATUS_CONNECTED, 0))
                .collect();
        }
        tracing::info!(
            kind = format_args!("{kind:#x}"),
            enumeration,
            devices = found.len(),
            "input device callback"
        );
        state.callbacks.push(registration);
        if enumeration == ASYNC_ENUMERATION {
            state.pending.append(&mut found);
        }
    }
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(registration.token) };
    }
    hub.deliver(found);
    S_OK
}
/// Guide button and keyboard layout callbacks: neither ever changes here.
extern "system" fn register_other(
    _: This,
    _device: *const c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    token(out)
}

fn forget(this: This, token: u64) {
    let mut state = hub(this).state();
    state.callbacks.retain(|r| r.token != token);
    state.pending.retain(|e| e.registration.token != token);
}
extern "system" fn stop_callback(this: This, token: u64) {
    forget(this, token);
}
/// Stops the callback, then waits up to `timeout` µs for a call of it that
/// is running; false on time-out, or from inside that callback (it cannot
/// wait for itself), as documented. Either way it is never called again.
extern "system" fn unregister_callback(this: This, token: u64, timeout: u64) -> bool {
    forget(this, token);
    let hub = hub(this);
    let me = std::thread::current().id();
    let deadline = Instant::now().checked_add(Duration::from_micros(timeout));
    let mut state = hub.state();
    loop {
        let running = state.running.iter().filter(|r| r.0 == token);
        if running.clone().any(|r| r.1 == me) {
            return false;
        }
        if running.count() == 0 {
            return true;
        }
        state = match deadline {
            None => hub
                .returned
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner),
            Some(deadline) => {
                let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                    return false;
                };
                let waited = hub.returned.wait_timeout(state, left);
                waited.unwrap_or_else(PoisonError::into_inner).0
            }
        };
    }
}

/// Dispatcher and aggregate device: no title of ours asks for them.
extern "system" fn create_dispatcher(_: This, out: Out) -> HRESULT {
    tracing::warn!("input dispatcher not implemented");
    fail(out, E_NOTIMPL)
}
extern "system" fn create_aggregate(_: This, _kind: u32, out: Out) -> HRESULT {
    tracing::warn!("aggregate input device not implemented");
    fail(out, E_NOTIMPL)
}
extern "system" fn find_by_id(this: This, id: *const [u8; 32], out: Out) -> HRESULT {
    if id.is_null() || out.is_null() {
        return fail(out, E_INVALIDARG);
    }
    // SAFETY: checked non-null; the title passes a device id.
    let id = unsafe { &*id };
    match hub(this).devices.iter().find(|d| d.info.id == *id) {
        Some(device) => {
            // SAFETY: checked non-null.
            unsafe { out.write((device as *const Device).cast()) };
            S_OK
        }
        None => fail(out, DEVICE_NOT_FOUND),
    }
}
/// Lookup by object, platform handle or platform string: our devices have
/// none of these.
extern "system" fn find_device(_: This, _value: *const c_void, out: Out) -> HRESULT {
    fail(out, DEVICE_NOT_FOUND)
}
extern "system" fn enable_oem_device(
    _: This,
    _vendor: u16,
    _product: u16,
    _interface: u8,
    _collection: u8,
) -> HRESULT {
    S_OK
}
extern "system" fn set_focus_policy(_: This, _policy: u32) {}

static INPUT: Table<22> = Table([
    no_interface as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    timestamp as *const (),           // 0x18
    current_reading as *const (),     // 0x20
    next_reading as *const (),        // 0x28
    previous_reading as *const (),    // 0x30
    temporal_reading as *const (),    // 0x38
    register_reading as *const (),    // 0x40
    register_device as *const (),     // 0x48
    register_other as *const (),      // 0x50 guide button
    register_other as *const (),      // 0x58 keyboard layout
    stop_callback as *const (),       // 0x60
    unregister_callback as *const (), // 0x68
    create_dispatcher as *const (),   // 0x70
    create_aggregate as *const (),    // 0x78
    find_by_id as *const (),          // 0x80
    find_device as *const (),         // 0x88 by object
    find_device as *const (),         // 0x90 by platform handle
    find_device as *const (),         // 0x98 by platform string
    enable_oem_device as *const (),   // 0xa0
    set_focus_policy as *const (),    // 0xa8
]);

extern "system" fn device_info(this: *const Device) -> *const DeviceInfo {
    // SAFETY: one of our devices.
    unsafe { &(*this).info }
}
extern "system" fn device_status(this: *const Device) -> u32 {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    if device.hub().state().slots[device.index].connected {
        STATUS_CONNECTED
    } else {
        0
    }
}
// shortcut: trigger motors are dropped (XInput has none). The host API that
// drives them without a redistributable is Windows.Gaming.Input (WinRT
// `Gamepad.Vibration`), which needs WinRT bindings this crate lacks outside
// diagnostics builds; Grand Theft Auto VI sends trigger levels in slot 0x50.
extern "system" fn set_rumble(this: *const Device, params: *const [f32; 4]) {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    if device.index >= PADS {
        return;
    }
    let [low, high, ..] = if params.is_null() {
        [0.0; 4]
    } else {
        // SAFETY: checked non-null; four motor levels.
        unsafe { *params }
    };
    let speed = |level: f32| (level.clamp(0.0, 1.0) * f32::from(u16::MAX)) as u16;
    lock(&device.hub().source)
        .input
        .rumble(device.index, speed(low), speed(high));
}
extern "system" fn unsupported<const SLOT: usize>(_: *const c_void) -> HRESULT {
    tracing::warn!(
        slot = format_args!("{SLOT:#x}"),
        "input device method not implemented"
    );
    E_NOTIMPL
}

static DEVICE: Table<21> = Table([
    no_interface as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    device_info as *const (),         // 0x18
    device_status as *const (),       // 0x20
    unsupported::<0x28> as *const (), // battery
    unsupported::<0x30> as *const (), // force feedback
    unsupported::<0x38> as *const (), //
    unsupported::<0x40> as *const (), //
    unsupported::<0x48> as *const (), // haptics
    set_rumble as *const (),          // 0x50
    unsupported::<0x58> as *const (), // input synchronization
    unsupported::<0x60> as *const (), //
    unsupported::<0x68> as *const (), // power off
    unsupported::<0x70> as *const (), // raw device access
    unsupported::<0x78> as *const (), //
    unsupported::<0x80> as *const (), //
    unsupported::<0x88> as *const (), //
    unsupported::<0x90> as *const (), //
    unsupported::<0x98> as *const (), //
    unsupported::<0xa0> as *const (), //
]);

fn reading<'a>(this: *const Reading) -> &'a Reading {
    // SAFETY: a reading we handed out and the title still holds.
    unsafe { &*this }
}
extern "system" fn reading_add_ref(this: *const Reading) -> u32 {
    reading(this).refs.fetch_add(1, Ordering::Relaxed) + 1
}
extern "system" fn reading_release(this: *const Reading) -> u32 {
    let left = reading(this).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if left == 0 {
        // SAFETY: the last reference is gone; `Ref::new` boxed it.
        drop(unsafe { Box::from_raw(this.cast_mut()) });
    }
    left
}
extern "system" fn input_kind(this: *const Reading) -> u32 {
    reading(this).kind()
}
extern "system" fn sequence_number(this: *const Reading, kind: u32) -> u64 {
    let reading = reading(this);
    if reading.kind() & kind != 0 {
        reading.sequence
    } else {
        0
    }
}
extern "system" fn reading_timestamp(this: *const Reading) -> u64 {
    reading(this).timestamp
}
extern "system" fn reading_device(this: *const Reading, out: *mut *const Device) {
    if !out.is_null() {
        // SAFETY: checked non-null; devices need no reference.
        unsafe { out.write(reading(this).device) };
    }
}
extern "system" fn key_count(this: *const Reading) -> u32 {
    match &reading(this).value {
        Value::Keys(keys) => keys.len() as u32,
        Value::Pad(_) => 0,
    }
}
extern "system" fn key_state(this: *const Reading, count: u32, out: *mut KeyState) -> u32 {
    let Value::Keys(keys) = &reading(this).value else {
        return 0;
    };
    if out.is_null() {
        return 0;
    }
    let n = keys.len().min(count as usize);
    // SAFETY: the title passes room for `count` keys.
    unsafe { std::ptr::copy_nonoverlapping(keys.as_ptr(), out, n) };
    n as u32
}
extern "system" fn gamepad_state(this: *const Reading, out: *mut PadState) -> bool {
    match &reading(this).value {
        Value::Pad(state) if !out.is_null() => {
            // SAFETY: checked non-null.
            unsafe { out.write(*state) };
            true
        }
        _ => false,
    }
}
/// Kinds our devices do not have: no count, no state.
extern "system" fn nothing(_: *const c_void) -> usize {
    0
}

static READING: Table<25> = Table([
    no_interface as *const (),      // 0x00
    reading_add_ref as *const (),   // 0x08
    reading_release as *const (),   // 0x10
    input_kind as *const (),        // 0x18
    sequence_number as *const (),   // 0x20
    reading_timestamp as *const (), // 0x28
    reading_device as *const (),    // 0x30
    nothing as *const (),           // 0x38 raw report
    nothing as *const (),           // 0x40 controller axes, buttons, switches
    nothing as *const (),           // 0x48
    nothing as *const (),           // 0x50
    nothing as *const (),           // 0x58
    nothing as *const (),           // 0x60
    nothing as *const (),           // 0x68
    key_count as *const (),         // 0x70
    key_state as *const (),         // 0x78
    nothing as *const (),           // 0x80 mouse
    nothing as *const (),           // 0x88 touch
    nothing as *const (),           // 0x90
    nothing as *const (),           // 0x98 motion
    nothing as *const (),           // 0xa0 arcade stick
    nothing as *const (),           // 0xa8 flight stick
    gamepad_state as *const (),     // 0xb0
    nothing as *const (),           // 0xb8 racing wheel
    nothing as *const (),           // 0xc0 UI navigation
]);

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "gameinput",
        dlls: &["gameinput.dll"],
        // Titles import it by ordinal only.
        exports: vec![HleExport {
            name: "#1",
            address: GameInputCreate as *const () as usize,
        }],
    }
}

/// Host input: XInput pads, the keyboard while our process has focus.
#[cfg(windows)]
mod host {
    use std::ptr::null_mut;
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};

    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
    use windows_sys::Win32::System::Threading::GetCurrentProcessId;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, MAPVK_VK_TO_VSC_EX, MapVirtualKeyW,
    };
    use windows_sys::Win32::UI::Input::XboxController::{
        XINPUT_GAMEPAD, XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_BACK,
        XINPUT_GAMEPAD_DPAD_DOWN, XINPUT_GAMEPAD_DPAD_LEFT, XINPUT_GAMEPAD_DPAD_RIGHT,
        XINPUT_GAMEPAD_DPAD_UP, XINPUT_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_LEFT_THUMB,
        XINPUT_GAMEPAD_RIGHT_SHOULDER, XINPUT_GAMEPAD_RIGHT_THUMB, XINPUT_GAMEPAD_START,
        XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y, XINPUT_STATE, XINPUT_VIBRATION, XInputGetState,
        XInputSetState,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        CallWindowProcW, ClipCursor, DefWindowProcW, GWLP_WNDPROC, GetClassNameW, GetCursorPos,
        GetForegroundWindow, GetPropW, GetWindowInfo, GetWindowLongPtrW, GetWindowThreadProcessId,
        HTCLIENT, RemovePropW, SendNotifyMessageW, SetCursor, SetCursorPos, SetPropW,
        SetWindowLongPtrW, WINDOWINFO, WM_MOUSEMOVE, WM_NCDESTROY, WM_SETCURSOR, WNDPROC,
    };
    use windows_sys::core::{PCWSTR, w};

    use super::*;

    /// Asking XInput about an empty slot is slow; look again this often.
    const EMPTY_SLOT_RECHECK: Duration = Duration::from_secs(1);

    const BUTTONS: [(u16, u32); 14] = [
        (XINPUT_GAMEPAD_START, MENU),
        (XINPUT_GAMEPAD_BACK, VIEW),
        (XINPUT_GAMEPAD_A, A),
        (XINPUT_GAMEPAD_B, B),
        (XINPUT_GAMEPAD_X, X),
        (XINPUT_GAMEPAD_Y, Y),
        (XINPUT_GAMEPAD_DPAD_UP, UP),
        (XINPUT_GAMEPAD_DPAD_DOWN, DOWN),
        (XINPUT_GAMEPAD_DPAD_LEFT, LEFT),
        (XINPUT_GAMEPAD_DPAD_RIGHT, RIGHT),
        (XINPUT_GAMEPAD_LEFT_SHOULDER, LEFT_SHOULDER),
        (XINPUT_GAMEPAD_RIGHT_SHOULDER, RIGHT_SHOULDER),
        (XINPUT_GAMEPAD_LEFT_THUMB, LEFT_STICK),
        (XINPUT_GAMEPAD_RIGHT_THUMB, RIGHT_STICK),
    ];

    /// Raw values, no dead zone: titles apply their own.
    pub(super) fn convert(pad: &XINPUT_GAMEPAD) -> PadState {
        let stick = |v: i16| (f32::from(v) / 32767.0).max(-1.0);
        PadState {
            buttons: BUTTONS
                .iter()
                .filter(|(host, _)| pad.wButtons & host != 0)
                .fold(0, |all, (_, bit)| all | bit),
            left_trigger: f32::from(pad.bLeftTrigger) / 255.0,
            right_trigger: f32::from(pad.bRightTrigger) / 255.0,
            left_x: stick(pad.sThumbLX),
            left_y: stick(pad.sThumbLY),
            right_x: stick(pad.sThumbRX),
            right_y: stick(pad.sThumbRY),
        }
    }

    #[derive(Default)]
    pub(super) struct Host {
        empty_until: [Option<Instant>; PADS],
        mouse: Mouse,
    }

    fn key_state(vk: u8) -> KeyState {
        KeyState {
            // SAFETY: plain query.
            scan_code: unsafe { MapVirtualKeyW(u32::from(vk), MAPVK_VK_TO_VSC_EX) },
            virtual_key: vk,
            // shortcut: no characters (code point 0); Grand Theft
            // Auto VI reads scan codes and virtual keys only.
            ..Default::default()
        }
    }

    impl Source for Host {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            if self.empty_until[slot].is_some_and(|until| Instant::now() < until) {
                return None;
            }
            let mut state = XINPUT_STATE::default();
            // SAFETY: room for the state.
            if unsafe { XInputGetState(slot as u32, &mut state) } != 0 {
                self.empty_until[slot] = Some(Instant::now() + EMPTY_SLOT_RECHECK);
                return None;
            }
            self.empty_until[slot] = None;
            Some(convert(&state.Gamepad))
        }

        fn keys(&mut self) -> Vec<KeyState> {
            let mut down = scripted();
            if foreground_window().is_some() {
                // Left/right modifier keys only, as devices report them; no
                // mouse buttons (below 0x08).
                let held: Vec<u8> = (0x08..=0xFE_u8)
                    .filter(|vk| !(0x10..=0x12).contains(vk) && !down.contains(vk))
                    // SAFETY: plain queries.
                    .filter(|&vk| unsafe { GetAsyncKeyState(i32::from(vk)) } < 0)
                    .collect();
                down.extend(held);
            }
            down.into_iter().map(key_state).collect()
        }

        fn fallback_pad(&mut self, keys: &[KeyState], active: bool) -> PadState {
            static GTA: LazyLock<bool> =
                LazyLock::new(|| std::env::var_os("VICTORY_KBM_GTA").is_some_and(|v| v == "1"));
            if !*GTA {
                return fallback(keys);
            }
            let mouse = self.mouse.poll(active.then(foreground_window).flatten());
            merge_pad(keyboard_pad(keys, GTA_BUTTONS, None), mouse)
        }

        #[cfg(any(test, feature = "diagnostics"))]
        fn scripted_keys(&mut self) -> Vec<KeyState> {
            #[cfg(feature = "diagnostics")]
            {
                scripted().into_iter().map(key_state).collect()
            }
            #[cfg(not(feature = "diagnostics"))]
            {
                Vec::new()
            }
        }

        fn rumble(&mut self, slot: usize, low: u16, high: u16) {
            let speeds = XINPUT_VIBRATION {
                wLeftMotorSpeed: low,
                wRightMotorSpeed: high,
            };
            // SAFETY: a valid vibration; an empty slot just fails.
            unsafe { XInputSetState(slot as u32, &speeds) };
        }
    }

    /// Pointer capture is confined to the foreground stand-in's client area.
    /// Screen-coordinate deltas work with Wine's Mac cursor driver. Read back
    /// every warp so rounding by the driver does not feed motion into the pad.
    #[derive(Default)]
    struct Mouse {
        window: usize,
        capture: Option<Capture>,
        toggle_down: bool,
        buttons: Option<[bool; 3]>,
        pressed_inside: [bool; 3],
        look: MouseLook,
        last: Option<Instant>,
    }

    struct Capture {
        bounds: [i32; 4],
        anchor: POINT,
    }

    impl Mouse {
        fn release(&mut self) {
            if self.capture.take().is_some() {
                CAPTURED_WINDOW.store(0, Ordering::Release);
                // SAFETY: frees the desktop cursor after this instance clipped it.
                unsafe { ClipCursor(std::ptr::null()) };
                refresh_cursor(self.window as HWND);
            }
            self.look = MouseLook::default();
            self.last = None;
        }

        fn poll(&mut self, window: Option<HWND>) -> PadState {
            let hwnd = window.unwrap_or(null_mut());
            if self.window != hwnd as usize {
                self.release();
                self.window = hwnd as usize;
                self.toggle_down = true;
                self.buttons = None;
                self.pressed_inside = [false; 3];
            }
            if hwnd.is_null() {
                return PadState::default();
            }

            // F2 and VK_OEM_3 (the backtick/tilde key on US layouts).
            let toggle = held(0x71) || held(0xc0);
            let was_down = std::mem::replace(&mut self.toggle_down, toggle);
            let toggle_edge = toggle && !was_down;
            let pause = held(0x1b) || held(b'P'.into());
            let Some((rect, position)) = client_and_cursor(hwnd) else {
                self.release();
                self.buttons = None;
                self.pressed_inside = [false; 3];
                return PadState::default();
            };
            let bounds = [rect.left, rect.top, rect.right, rect.bottom];
            if pause || (toggle_edge && self.capture.is_some()) {
                self.release();
            } else if toggle_edge && install_cursor_proc(hwnd) {
                // SAFETY: clip to this foreground window's screen-space client rectangle.
                if unsafe { GetForegroundWindow() == hwnd && ClipCursor(&rect) != 0 } {
                    CAPTURED_WINDOW.store(self.window, Ordering::Release);
                    self.capture = Some(Capture {
                        bounds,
                        anchor: position,
                    });
                    if self.recenter(rect) {
                        refresh_cursor(hwnd);
                    }
                }
            } else if let Some(capture) = &mut self.capture {
                if capture.bounds != bounds {
                    // SAFETY: the client rectangle changed while it was captured.
                    if unsafe { ClipCursor(&rect) } == 0 {
                        self.release();
                    } else {
                        capture.bounds = bounds;
                        self.look = MouseLook::default();
                        self.recenter(rect);
                    }
                } else {
                    let delta = [position.x - capture.anchor.x, position.y - capture.anchor.y];
                    if delta == [0, 0] || self.recenter(rect) {
                        let now = Instant::now();
                        let elapsed = self.last.replace(now).map_or(POLL, |last| now - last);
                        self.look
                            .step(delta, elapsed, *MOUSE_SENSITIVITY, *MOUSE_INVERT_Y);
                    }
                }
            }

            let down = [held(0x01), held(0x02), held(0x04)];
            let in_client = (rect.left..rect.right).contains(&position.x)
                && (rect.top..rect.bottom).contains(&position.y);
            let previous = self.buttons.replace(down).unwrap_or(down);
            for ((pressed, held), was_held) in
                self.pressed_inside.iter_mut().zip(down).zip(previous)
            {
                *pressed =
                    held && (*pressed || (!was_held && (in_client || self.capture.is_some())));
            }
            self.look.pad(self.pressed_inside)
        }

        fn recenter(&mut self, rect: RECT) -> bool {
            let x = rect.left + (rect.right - rect.left) / 2;
            let y = rect.top + (rect.bottom - rect.top) / 2;
            let mut position = POINT::default();
            // SAFETY: screen coordinates in the client area and a valid output pointer.
            if unsafe {
                GetForegroundWindow() == self.window as HWND
                    && SetCursorPos(x, y) != 0
                    && GetCursorPos(&mut position) != 0
            } {
                if let Some(capture) = &mut self.capture {
                    capture.anchor = position;
                }
                true
            } else {
                self.release();
                false
            }
        }
    }

    static MOUSE_SENSITIVITY: LazyLock<f32> = LazyLock::new(|| {
        std::env::var("VICTORY_MOUSE_SENS")
            .ok()
            .and_then(|value| value.trim().parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(1.0)
    });
    static MOUSE_INVERT_Y: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("VICTORY_MOUSE_INVERT_Y").is_some_and(|v| v == "1"));

    fn held(vk: i32) -> bool {
        // SAFETY: plain key state query.
        unsafe { GetAsyncKeyState(vk) < 0 }
    }

    fn client_and_cursor(hwnd: HWND) -> Option<(RECT, POINT)> {
        let mut info = WINDOWINFO {
            cbSize: size_of::<WINDOWINFO>() as u32,
            ..Default::default()
        };
        let mut position = POINT::default();
        // SAFETY: both output structures are initialized and correctly sized.
        let ok = unsafe { GetWindowInfo(hwnd, &mut info) != 0 && GetCursorPos(&mut position) != 0 };
        let rect = info.rcClient;
        (ok && rect.right > rect.left && rect.bottom > rect.top).then_some((rect, position))
    }

    static CAPTURED_WINDOW: AtomicUsize = AtomicUsize::new(0);
    const CURSOR_PROC: PCWSTR = w!("victory.gameinput.cursor-proc");

    /// SetCursor must run on the thread that owns the visible window. Chain
    /// its procedure only for our own stand-in, whose procedure stays fixed.
    fn install_cursor_proc(hwnd: HWND) -> bool {
        let mut class = [0; 32];
        let stand_in = w!("VIctory.title");
        // SAFETY: `hwnd` belongs to this process and the buffers are valid.
        unsafe {
            let len = GetClassNameW(hwnd, class.as_mut_ptr(), class.len() as i32) as usize;
            if len != 13 || class[..len] != *std::slice::from_raw_parts(stand_in, len) {
                return false;
            }
            if !GetPropW(hwnd, CURSOR_PROC).is_null() {
                return true;
            }
            let original = GetWindowLongPtrW(hwnd, GWLP_WNDPROC);
            // Publish the original before messages can reach the new procedure.
            if original == 0 || SetPropW(hwnd, CURSOR_PROC, original as _) == 0 {
                return false;
            }
            if SetWindowLongPtrW(hwnd, GWLP_WNDPROC, cursor_proc as *const () as isize) == 0 {
                RemovePropW(hwnd, CURSOR_PROC);
                return false;
            }
            true
        }
    }

    extern "system" fn cursor_proc(hwnd: HWND, message: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        // SAFETY: called by the window manager for our subclassed stand-in.
        unsafe {
            if message == WM_SETCURSOR
                && lp as u16 as u32 == HTCLIENT
                && CAPTURED_WINDOW.load(Ordering::Acquire) == hwnd as usize
                && GetForegroundWindow() == hwnd
            {
                SetCursor(null_mut());
                return 1;
            }
            let original = if message == WM_NCDESTROY {
                RemovePropW(hwnd, CURSOR_PROC)
            } else {
                GetPropW(hwnd, CURSOR_PROC)
            };
            if original.is_null() {
                return DefWindowProcW(hwnd, message, wp, lp);
            }
            let original = std::mem::transmute::<*mut c_void, WNDPROC>(original);
            CallWindowProcW(original, hwnd, message, wp, lp)
        }
    }

    fn refresh_cursor(hwnd: HWND) {
        // SAFETY: notify the owning UI thread asynchronously to avoid a wait in the poll.
        unsafe {
            if GetForegroundWindow() == hwnd {
                SendNotifyMessageW(
                    hwnd,
                    WM_SETCURSOR,
                    hwnd as usize,
                    HTCLIENT as isize | (WM_MOUSEMOVE as isize) << 16,
                );
            }
        }
    }

    /// Debugging aid: `VICTORY_KEYS=<seconds>:<key>,...` holds each key for
    /// 0.2 s that long after input starts, with or without focus. A key is a
    /// letter, a digit, or `Enter`, `Esc`, `Tab`, `Back`, `Up`, `Down`,
    /// `Left`, `Right` (the stand-in pad's table maps them to buttons).
    /// In every build, so that unattended runs can get past the title's
    /// prompts; unset, it holds nothing.
    fn scripted() -> Vec<u8> {
        static SCRIPT: LazyLock<(Instant, Vec<(Duration, u8)>)> = LazyLock::new(|| {
            let list = std::env::var("VICTORY_KEYS").unwrap_or_default();
            let keys = list.split(',').filter_map(|entry| {
                let (at, key) = entry.split_once(':')?;
                let vk = match key.trim() {
                    "Enter" => 0x0D,
                    "Esc" => 0x1B,
                    "Tab" => 0x09,
                    "Back" => 0x08,
                    "Up" => 0x26,
                    "Down" => 0x28,
                    "Left" => 0x25,
                    "Right" => 0x27,
                    k if k.len() == 1 && k.as_bytes()[0].is_ascii_alphanumeric() => {
                        k.as_bytes()[0].to_ascii_uppercase()
                    }
                    _ => return None,
                };
                Some((
                    Duration::try_from_secs_f64(at.trim().parse().ok()?).ok()?,
                    vk,
                ))
            });
            (Instant::now(), keys.collect())
        });
        let (start, keys) = &*SCRIPT;
        let now = start.elapsed();
        keys.iter()
            .filter(|&&(at, _)| (at..at + Duration::from_millis(200)).contains(&now))
            .map(|&(_, vk)| vk)
            .collect()
    }

    /// The foreground window is one of ours (the title's or our fallback).
    fn foreground_window() -> Option<HWND> {
        let mut process = 0;
        // SAFETY: plain queries; a null window gives process 0.
        unsafe {
            let hwnd = GetForegroundWindow();
            GetWindowThreadProcessId(hwnd, &mut process);
            (process == GetCurrentProcessId()).then_some(hwnd)
        }
    }

    #[test]
    fn host_pads_convert_to_console_pads() {
        let pad = XINPUT_GAMEPAD {
            wButtons: XINPUT_GAMEPAD_A | XINPUT_GAMEPAD_START | XINPUT_GAMEPAD_DPAD_LEFT,
            bLeftTrigger: 255,
            sThumbLX: -32768,
            sThumbLY: 32767,
            ..Default::default()
        };
        let state = convert(&pad);
        assert_eq!(state.buttons, A | MENU | LEFT);
        assert_eq!(
            (
                state.left_trigger,
                state.left_x,
                state.left_y,
                state.right_x
            ),
            (1.0, -1.0, 1.0, 0.0)
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::HleResolver;
    use crate::HostPolicy;
    use victory_loader::Provenance;
    use victory_pe::ImportSymbol;

    #[derive(Default)]
    struct FakeHost {
        pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
        scripted: Vec<KeyState>,
        gta_mouse: Option<PadState>,
        stand_in_active: bool,
        rumble: Vec<(usize, u16, u16)>,
    }

    struct Fake(Arc<Mutex<FakeHost>>);

    impl Source for Fake {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            self.0.lock().unwrap().pads[slot]
        }
        fn keys(&mut self) -> Vec<KeyState> {
            self.0.lock().unwrap().keys.clone()
        }
        fn fallback_pad(&mut self, keys: &[KeyState], active: bool) -> PadState {
            let mut host = self.0.lock().unwrap();
            host.stand_in_active = active;
            host.gta_mouse.map_or_else(
                || fallback(keys),
                |mouse| merge_pad(keyboard_pad(keys, GTA_BUTTONS, None), mouse),
            )
        }
        fn scripted_keys(&mut self) -> Vec<KeyState> {
            self.0.lock().unwrap().scripted.clone()
        }
        fn rumble(&mut self, slot: usize, low: u16, high: u16) {
            self.0.lock().unwrap().rumble.push((slot, low, high));
        }
    }

    fn fake() -> (&'static Hub, Arc<Mutex<FakeHost>>) {
        let host = Arc::new(Mutex::new(FakeHost::default()));
        let hub = Hub::new(Box::new(Fake(Arc::clone(&host))));
        hub.poll();
        (hub, host)
    }

    fn key(virtual_key: u8) -> KeyState {
        KeyState {
            scan_code: u32::from(virtual_key) + 0x100,
            virtual_key,
            ..Default::default()
        }
    }

    fn slot<F: Copy, T>(object: *const T, offset: usize) -> F {
        // SAFETY: our objects start with their table, which has the slot.
        let f = unsafe { *(*(object as *const *const *const ())).add(offset / 8) };
        // SAFETY: the caller names the slot's signature.
        unsafe { std::mem::transmute_copy(&f) }
    }

    /// Device callbacks seen: device index, current and previous status.
    type Log = Mutex<Vec<(usize, u32, u32)>>;

    extern "system" fn record(
        _token: u64,
        context: *mut c_void,
        device: *const Device,
        _timestamp: u64,
        current: u32,
        previous: u32,
    ) {
        // SAFETY: the test passes its log and one of the hub's devices.
        let (log, index) = unsafe { (&*(context as *const Log), (*device).index) };
        log.lock().unwrap().push((index, current, previous));
    }

    fn register(hub: &Hub, kind: u32, enumeration: u32, log: &Log) -> u64 {
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let mut token = 0;
        let context = log as *const Log as *mut c_void;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let hr = register(
            &hub.object,
            std::ptr::null(),
            kind,
            1,
            enumeration,
            context,
            Some(record),
            &mut token,
        );
        assert_eq!(hr, S_OK);
        assert_ne!(token, 0);
        token
    }

    fn result(hr: HRESULT, out: *const c_void) -> Result<*const Reading, HRESULT> {
        if hr == S_OK {
            Ok(out.cast())
        } else {
            assert!(out.is_null());
            Err(hr)
        }
    }

    fn current(hub: &Hub, kind: u32, device: usize) -> Result<*const Reading, HRESULT> {
        type Current = extern "system" fn(This, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let hr = slot::<Current, _>(&hub.object, 0x20)(
            &hub.object,
            kind,
            &hub.devices[device],
            &mut out,
        );
        result(hr, out)
    }

    fn walk(hub: &Hub, offset: usize, from: *const Reading) -> Result<*const Reading, HRESULT> {
        type Walk = extern "system" fn(This, *const Reading, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let device = reading(from).device;
        let hr =
            slot::<Walk, _>(&hub.object, offset)(&hub.object, from, GAMEPAD_KIND, device, &mut out);
        result(hr, out)
    }

    fn pad(reading: *const Reading) -> PadState {
        let mut state = PadState::default();
        let get =
            slot::<extern "system" fn(*const Reading, *mut PadState) -> bool, _>(reading, 0xb0);
        assert!(get(reading, &mut state));
        state
    }

    fn stamp(reading: *const Reading) -> (u64, u64) {
        let time = slot::<extern "system" fn(*const Reading) -> u64, _>(reading, 0x28);
        let sequence = slot::<extern "system" fn(*const Reading, u32) -> u64, _>(reading, 0x20);
        (time(reading), sequence(reading, GAMEPAD_KIND))
    }

    fn release(reading: *const Reading) {
        slot::<extern "system" fn(*const Reading) -> u32, _>(reading, 0x10)(reading);
    }

    #[test]
    fn ordinal_one_resolves_to_create() {
        let r = HleResolver::new(HostPolicy::none());
        assert_eq!(
            r.classify("GameInput.dll", &ImportSymbol::Ordinal(1)),
            Some(Provenance::Hle)
        );
        assert_eq!(r.classify("GameInput.dll", &ImportSymbol::Ordinal(2)), None);
    }

    #[test]
    fn pads_connect_report_and_disconnect() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, GAMEPAD_KIND, 2, &log);
        // Blocking enumeration ran before registration returned: with no
        // host pad the keyboard's stand-in is the only one.
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        host.lock().unwrap().pads[1] = Some(PadState {
            buttons: A,
            left_x: 0.5,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[1..],
            [(1, STATUS_CONNECTED, 0), (FALLBACK, 0, STATUS_CONNECTED)]
        );

        let first = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(first).buttons, A);
        assert_eq!(pad(first).left_x, 0.5);
        // No change, no new reading.
        hub.poll();
        let again = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(again, first);
        release(again);

        std::thread::sleep(Duration::from_millis(2));
        host.lock().unwrap().pads[1].as_mut().unwrap().buttons = B;
        hub.poll();
        let second = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(second).buttons, B);
        let (t1, s1) = stamp(first);
        let (t2, s2) = stamp(second);
        assert!(t2 > t1 && s2 == s1 + 1);
        assert!(t2 <= now());

        // The title's edge check: previous of current, and back again.
        assert_eq!(walk(hub, 0x30, second), Ok(first));
        assert_eq!(walk(hub, 0x28, first), Ok(second));
        assert_eq!(walk(hub, 0x28, second), Err(READING_NOT_FOUND));
        assert_eq!(walk(hub, 0x30, first), Err(READING_NOT_FOUND));
        for r in [first, first, second, second] {
            release(r);
        }

        host.lock().unwrap().pads[1] = None;
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[3..],
            [(1, 0, STATUS_CONNECTED), (FALLBACK, STATUS_CONNECTED, 0)]
        );
        assert_eq!(current(hub, GAMEPAD_KIND, 1), Err(DEVICE_DISCONNECTED));
        // The stand-in is back, at rest.
        let rest = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(pad(rest), PadState::default());
        release(rest);
    }

    #[test]
    fn with_no_device_readings_are_walked_across_devices_in_time_order() {
        type Walk = extern "system" fn(This, *const Reading, u32, *const Device, Out) -> HRESULT;
        let any = |hub: &Hub, offset, from, kind| {
            let mut out = std::ptr::null();
            let walk = slot::<Walk, _>(&hub.object, offset);
            result(
                walk(&hub.object, from, kind, std::ptr::null(), &mut out),
                out,
            )
        };
        let both = GAMEPAD_KIND | KEYBOARD_KIND;
        let tick = || std::thread::sleep(Duration::from_millis(2));
        let (hub, host) = fake();
        // The first poll: the stand-in pad and the keyboard, at one time.
        let stand_in = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        let no_keys = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        tick();
        host.lock().unwrap().pads[1] = Some(PadState {
            buttons: A,
            ..Default::default()
        });
        hub.poll();
        let pad_a = current(hub, GAMEPAD_KIND, 1).unwrap();
        tick();
        host.lock().unwrap().keys = vec![key(b'W')];
        hub.poll();
        let w = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        tick();
        host.lock().unwrap().pads[1].as_mut().unwrap().buttons = B;
        hub.poll();
        let pad_b = current(hub, GAMEPAD_KIND, 1).unwrap();

        let order = [stand_in, no_keys, pad_a, w, pad_b];
        for pair in order.windows(2) {
            assert_eq!(any(hub, 0x28, pair[0], both), Ok(pair[1]));
            assert_eq!(any(hub, 0x30, pair[1], both), Ok(pair[0]));
        }
        assert_eq!(any(hub, 0x30, stand_in, both), Err(READING_NOT_FOUND));
        assert_eq!(any(hub, 0x28, pad_b, both), Err(READING_NOT_FOUND));
        // Only gamepads: the keyboard's readings are skipped.
        assert_eq!(any(hub, 0x28, stand_in, GAMEPAD_KIND), Ok(pad_a));
        assert_eq!(any(hub, 0x28, pad_a, GAMEPAD_KIND), Ok(pad_b));
        assert_eq!(any(hub, 0x30, pad_a, GAMEPAD_KIND), Ok(stand_in));
        // Naming the device keeps the walk on it.
        assert_eq!(walk(hub, 0x30, pad_a), Err(READING_NOT_FOUND));
        for r in order {
            release(r);
        }
    }

    #[test]
    fn keys_feed_the_keyboard_and_the_stand_in_pad() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, KEYBOARD_KIND, 2, &log);
        assert_eq!(*log.lock().unwrap(), [(KEYBOARD, STATUS_CONNECTED, 0)]);

        let keys = vec![key(b'W'), key(0x0D), key(b'L'), key(b'C')];
        host.lock().unwrap().keys = keys.clone();
        hub.poll();
        let r = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        let count = slot::<extern "system" fn(*const Reading) -> u32, _>(r, 0x70);
        let state =
            slot::<extern "system" fn(*const Reading, u32, *mut KeyState) -> u32, _>(r, 0x78);
        assert_eq!(count(r), 4);
        let mut got = [KeyState::default(); 8];
        assert_eq!(state(r, 8, got.as_mut_ptr()), 4);
        assert_eq!(got[..4], keys[..]);
        release(r);

        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(
            pad(r),
            PadState {
                buttons: A,
                left_y: 1.0,
                right_x: 1.0,
                right_trigger: 1.0,
                ..Default::default()
            }
        );
        release(r);
        // A keyboard reading is no gamepad reading.
        assert_eq!(current(hub, GAMEPAD_KIND, KEYBOARD), Err(READING_NOT_FOUND));
    }

    #[test]
    fn gta_keyboard_and_mouse_feed_the_stand_in_pad_until_a_controller_takes_over() {
        let (hub, host) = fake();
        let mut look = MouseLook::default();
        look.step([50, -25], POLL, 1.0, false);
        {
            let mut host = host.lock().unwrap();
            host.keys = [
                b'W', 0xA1, 0x20, b'F', b'R', 0x09, b'Q', 0xA2, b'E', b'V', b'Z',
            ]
            .into_iter()
            .map(key)
            .collect();
            host.gta_mouse = Some(look.pad([true, true, true]));
            host.pads[0] = Some(PadState::default());
        }
        hub.poll();
        let reading = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        let state = pad(reading);
        release(reading);
        assert_eq!(
            state.buttons,
            A | X
                | Y
                | B
                | LEFT_SHOULDER
                | RIGHT_SHOULDER
                | LEFT_STICK
                | RIGHT_STICK
                | RIGHT
                | VIEW
                | DOWN
        );
        assert_eq!(
            (state.left_y, state.right_x, state.right_y),
            (1.0, 1.0, 0.55)
        );
        assert_eq!((state.left_trigger, state.right_trigger), (1.0, 1.0));
        assert!(host.lock().unwrap().stand_in_active);

        host.lock().unwrap().pads[0].as_mut().unwrap().buttons = A;
        hub.poll();
        assert!(!host.lock().unwrap().stand_in_active);
        assert_eq!(
            current(hub, GAMEPAD_KIND, FALLBACK),
            Err(DEVICE_DISCONNECTED)
        );
    }

    #[test]
    fn gta_mouse_look_inverts_and_decays_to_neutral() {
        let mut look = MouseLook::default();
        look.step([1, 10], POLL, 1.0, true);
        let motion = look.pad([true, false, false]);
        assert_eq!(motion.right_x, 0.022);
        assert!(motion.right_y > 0.0);
        assert_eq!((motion.left_trigger, motion.right_trigger), (0.0, 1.0));
        look.step([0, 0], Duration::from_millis(45), 1.0, true);
        let decayed = look.pad([false; 3]);
        assert_eq!(decayed.right_y, motion.right_y * 0.5);
        look.step([0, 0], Duration::from_secs(1), 1.0, true);
        assert_eq!(look.pad([false; 3]), PadState::default());
    }

    #[test]
    fn async_enumeration_waits_for_the_poll_and_unregistering_stops_callbacks() {
        let (hub, host) = fake();
        let log = Log::default();
        let token = register(hub, GAMEPAD_KIND, ASYNC_ENUMERATION, &log);
        assert!(log.lock().unwrap().is_empty());
        hub.poll();
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(unregister(&hub.object, token, 5000));
        host.lock().unwrap().pads[0] = Some(PadState {
            buttons: A,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(log.lock().unwrap().len(), 1);
    }

    #[test]
    fn unregistering_waits_for_a_running_callback() {
        use std::sync::atomic::AtomicBool;
        static INSIDE: AtomicBool = AtomicBool::new(false);
        static LEAVE: AtomicBool = AtomicBool::new(false);
        static FROM_INSIDE: AtomicU32 = AtomicU32::new(0);
        extern "system" fn slow(
            token: u64,
            context: *mut c_void,
            _: *const Device,
            _: u64,
            _: u32,
            _: u32,
        ) {
            // From inside its own callback, unregistering fails at once.
            let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(context, 0x68);
            if !unregister(context.cast(), token, 0) {
                FROM_INSIDE.fetch_add(1, Ordering::SeqCst);
            }
            INSIDE.store(true, Ordering::SeqCst);
            while !LEAVE.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let (hub, host) = fake();
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let mut token = 0;
        let context = (&raw const hub.object).cast_mut().cast();
        assert_eq!(
            register(
                &hub.object,
                std::ptr::null(),
                GAMEPAD_KIND,
                1,
                0,
                context,
                Some(slow),
                &mut token
            ),
            S_OK
        );
        host.lock().unwrap().pads[1] = Some(PadState {
            buttons: A,
            ..Default::default()
        });
        let poller = std::thread::spawn(|| hub.poll());
        while !INSIDE.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(
            !unregister(&hub.object, token, 5000),
            "times out while it runs"
        );
        let leave = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            LEAVE.store(true, Ordering::SeqCst);
        });
        let start = Instant::now();
        assert!(unregister(&hub.object, token, u64::MAX), "waits for it");
        assert!(start.elapsed() >= Duration::from_millis(40));
        assert_eq!(FROM_INSIDE.load(Ordering::SeqCst), 1);
        poller.join().unwrap();
        leave.join().unwrap();
    }

    #[test]
    fn devices_are_found_by_id_and_rumble_reaches_the_host() {
        let (hub, host) = fake();
        host.lock().unwrap().pads[2] = Some(PadState::default());
        hub.poll();
        let device = &hub.devices[2] as *const Device;
        let info = slot::<extern "system" fn(*const Device) -> *const u8, _>(device, 0x18);
        // SAFETY: the info has the id at +0x20.
        let id = unsafe { *(info(device).add(0x20) as *const [u8; 32]) };
        assert_ne!(id, [0; 32]);

        type Find = extern "system" fn(This, *const [u8; 32], Out) -> HRESULT;
        let find = slot::<Find, _>(&hub.object, 0x80);
        let mut out = std::ptr::null();
        assert_eq!(find(&hub.object, &id, &mut out), S_OK);
        assert_eq!(out, device.cast());
        assert_eq!(find(&hub.object, &[1; 32], &mut out), DEVICE_NOT_FOUND);
        assert!(out.is_null());

        type Rumble = extern "system" fn(*const Device, *const [f32; 4]);
        let rumble = slot::<Rumble, _>(device, 0x50);
        rumble(device, &[0.5, 1.0, 1.0, 1.0]);
        rumble(&hub.devices[FALLBACK], &[1.0; 4]);
        rumble(device, std::ptr::null());
        assert_eq!(host.lock().unwrap().rumble, [(2, 32767, 65535), (2, 0, 0)]);
    }

    #[test]
    fn scripted_keys_merge_into_connected_host_pad() {
        let (hub, host) = fake();
        let physical = PadState {
            buttons: B,
            left_x: -0.5,
            ..Default::default()
        };
        {
            let mut h = host.lock().unwrap();
            h.pads[1] = Some(physical);
            h.scripted = vec![key(0x0D), key(b'W'), key(b'C')];
        }
        hub.poll();

        assert_eq!(current(hub, GAMEPAD_KIND, FALLBACK), Err(DEVICE_DISCONNECTED));
        let r = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(
            pad(r),
            PadState {
                buttons: B | A,
                left_x: -0.5,
                left_y: 1.0,
                right_trigger: 1.0,
                ..Default::default()
            }
        );
        release(r);

        host.lock().unwrap().scripted.clear();
        hub.poll();
        let r = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(r), physical);
        release(r);
    }

    #[test]
    fn switched_off_receiver_does_not_hide_the_keyboard_stand_in() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, GAMEPAD_KIND, 2, &log);
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        // A receiver answering with an all-zero state while its pad is off
        // stays disconnected and leaves the keyboard stand-in active.
        host.lock().unwrap().pads[0] = Some(PadState::default());
        hub.poll();
        assert_eq!(log.lock().unwrap().len(), 1);
        assert_eq!(current(hub, GAMEPAD_KIND, 0), Err(DEVICE_DISCONNECTED));

        host.lock().unwrap().keys = vec![key(0x0D), key(b'W')];
        hub.poll();
        let stand_in = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(
            pad(stand_in),
            PadState {
                buttons: A,
                left_y: 1.0,
                ..Default::default()
            }
        );
        release(stand_in);

        // Scripted diagnostic keys drive the stand-in without latching the
        // idle receiver slot as an active physical controller.
        {
            let mut h = host.lock().unwrap();
            h.keys.clear();
            h.scripted = vec![key(b'X')];
        }
        hub.poll();
        assert_eq!(current(hub, GAMEPAD_KIND, 0), Err(DEVICE_DISCONNECTED));
        let scripted_r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(pad(scripted_r).buttons, X);
        release(scripted_r);

        host.lock().unwrap().scripted.clear();
        hub.poll();
        let rest = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(pad(rest), PadState::default());
        release(rest);

        // Turning the pad on and pressing a button connects slot 0, drops the
        // stand-in, and preserves that first press as the current reading.
        host.lock().unwrap().pads[0] = Some(PadState {
            buttons: B,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[1..],
            [(0, STATUS_CONNECTED, 0), (FALLBACK, 0, STATUS_CONNECTED)]
        );
        assert_eq!(
            current(hub, GAMEPAD_KIND, FALLBACK),
            Err(DEVICE_DISCONNECTED)
        );
        let first = current(hub, GAMEPAD_KIND, 0).unwrap();
        assert_eq!(pad(first).buttons, B);
        release(first);

        // Releasing the button back to rest keeps slot 0 connected.
        host.lock().unwrap().pads[0] = Some(PadState::default());
        hub.poll();
        let idle = current(hub, GAMEPAD_KIND, 0).unwrap();
        assert_eq!(pad(idle), PadState::default());
        release(idle);

        // Unplugging clears `seen_input` so a later neutral receiver on the
        // same slot does not take the stand-in away again.
        host.lock().unwrap().pads[0] = None;
        hub.poll();
        assert_eq!(current(hub, GAMEPAD_KIND, 0), Err(DEVICE_DISCONNECTED));

        {
            let mut h = host.lock().unwrap();
            h.pads[0] = Some(PadState::default());
            h.keys = vec![key(0x0D)];
        }
        hub.poll();
        assert_eq!(current(hub, GAMEPAD_KIND, 0), Err(DEVICE_DISCONNECTED));
        let back = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(pad(back).buttons, A);
        release(back);
    }
}
