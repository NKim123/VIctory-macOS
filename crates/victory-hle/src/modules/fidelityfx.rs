//! `amd_fidelityfx_loader_xbox.dll`: AMD FidelityFX's API, served by the
//! host's PC build of it on the host GPU when the user names a folder that
//! holds one (`VICTORY_FIDELITYFX_DIR`). Without it, or if it offers no
//! upscaler for the host GPU, this module stays out of the registry and the
//! title's own console DLLs run as before.
//!
//! Both builds have the same API (the FidelityFX SDK's, MIT): five functions
//! over chains of descriptions, each starting with a type id and the next
//! one's address; contexts are opaque pointers. The console-specific parts
//! are translated: the backend description names the console device (type
//! 0x0100_0002; on PC type 2 with the host device), and a dispatch names the
//! title's command list and resources (their host list and host copies, in
//! the states the title gives, see `victory_gpu::external`). The rest passes
//! through unchanged: sizes, flags, jitter, motion-vector scale, camera
//! values, the resource descriptions, the context. Only the upscaler: other
//! effects (frame generation) are refused.
#![allow(non_snake_case)]
// Title code calls these through its import table.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::BTreeMap;
use std::ffi::{CStr, c_char, c_void};
use std::path::Path;
use std::ptr::{null, null_mut};
use std::sync::{Arc, Mutex, OnceLock};

use victory_gpu::object::Object;
use windows::Win32::Graphics::Direct3D12::ID3D12Device;
use windows::core::Interface;

use crate::{HleModule, hle_export};

/// The PC build's upscaler, which has the five functions itself.
const DLLS: [&str; 2] = [
    "amd_fidelityfx_upscaler_dx12.dll",
    "amd_fidelityfx_dx12.dll",
];

// Return codes.
const OK: u32 = 0;
const UNKNOWN_TYPE: u32 = 2;
const NO_PROVIDER: u32 = 4;
const PARAMETER: u32 = 6;

// Description types.
const CREATE_UPSCALE: u64 = 0x1_0000;
const DISPATCH_UPSCALE: u64 = 0x1_0001;
const CONSOLE_BACKEND: u64 = 0x100_0002;
const HOST_BACKEND: u64 = 2;
const GET_VERSIONS: u64 = 4;
const GET_PROVIDER_VERSION: u64 = 6;
const MEMORY_USAGE_WITH_DEVICE: u64 = 0x1_0009;
const DISPLAY_RESOLUTION_MOTION_VECTORS: u32 = 1 << 1;

#[repr(C)]
#[derive(Clone, Copy)]
struct Header {
    kind: u64,
    next: *mut Header,
}

/// An upscaler context: flags, largest render and output sizes, a message
/// callback.
#[repr(C)]
#[derive(Clone, Copy)]
struct CreateUpscale {
    header: Header,
    flags: u32,
    max_render: [u32; 2],
    max_output: [u32; 2],
    message: usize,
}

/// The backend description: a device.
#[repr(C)]
struct Backend {
    header: Header,
    device: *mut c_void,
}

/// A resource, its description (type, format, sizes, mips, flags, usage)
/// and the state it is in.
#[repr(C)]
#[derive(Clone, Copy)]
struct Resource {
    resource: *mut c_void,
    description: [u32; 8],
    state: u32,
}

/// One upscaling: the command list, then colour, depth, motion vectors,
/// exposure, reactive mask, transparency mask and output; then jitter,
/// motion-vector scale, sizes, sharpening, frame time, pre-exposure, reset,
/// camera and flags, which pass through.
#[repr(C)]
#[derive(Clone, Copy)]
struct DispatchUpscale {
    header: Header,
    list: *mut c_void,
    resources: [Resource; 7],
    jitter: [f32; 2],
    motion_scale: [f32; 2],
    render_size: [u32; 2],
    upscale_size: [u32; 2],
    settings: [u32; 10],
}

const _: () = assert!(size_of::<CreateUpscale>() == 0x30);
const _: () = assert!(size_of::<Resource>() == 0x30);
const _: () = assert!(size_of::<DispatchUpscale>() == 0x1b0);

/// The versions a backend offers: count in and out, ids, names.
#[repr(C)]
#[derive(Clone, Copy)]
struct GetVersions {
    header: Header,
    create_kind: u64,
    device: *mut c_void,
    count: *mut u64,
    ids: *mut u64,
    names: *mut *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryUsageWithDevice {
    header: Header,
    device: *mut c_void,
    max_render: [u32; 2],
    max_output: [u32; 2],
    flags: u32,
    usage: *mut c_void,
}

#[repr(C)]
struct ProviderVersion {
    header: Header,
    id: u64,
    name: *const c_char,
}

type Create = unsafe extern "C" fn(*mut *mut c_void, *mut Header, *const c_void) -> u32;
type Destroy = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> u32;
/// Configure, query and dispatch.
type Call = unsafe extern "C" fn(*mut *mut c_void, *mut Header) -> u32;

struct Host {
    _library: Library,
    create: Create,
    destroy: Destroy,
    configure: Call,
    query: Call,
    dispatch: Call,
    device: ID3D12Device,
}

struct Library(windows_sys::Win32::Foundation::HMODULE);

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: this owns the reference acquired by LoadLibraryExW.
        unsafe {
            windows_sys::Win32::Foundation::FreeLibrary(self.0);
        }
    }
}

// SAFETY: D3D12 devices are free-threaded; the rest is code.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

static HOST: OnceLock<Host> = OnceLock::new();
struct Context {
    // Serializes recording and destruction for this particular temporal history.
    live: Mutex<bool>,
    // The FFX API requires pointers in the creation descriptor to remain
    // live through destroy, including our replacement backend.
    creation: CreateChain,
}

struct CreateChain {
    description: Box<CreateUpscale>,
    _backend: Box<Backend>,
}

// SAFETY: after Create, both boxed descriptors are retained immutably.
// The device is held by Host; the remaining extension pointers belong
// to the title under the FFX context lifetime contract.
unsafe impl Send for CreateChain {}
// SAFETY: as above, no thread mutates the retained descriptors.
unsafe impl Sync for CreateChain {}

static CONTEXTS: Mutex<BTreeMap<usize, Arc<Context>>> = Mutex::new(BTreeMap::new());

fn fits(extent: [u32; 2], maximum: [u32; 2]) -> bool {
    extent[0] != 0 && extent[1] != 0 && extent[0] <= maximum[0] && extent[1] <= maximum[1]
}

fn dispatch_output(
    render: [u32; 2],
    output: [u32; 2],
    max_render: [u32; 2],
    max_output: [u32; 2],
) -> Option<[u32; 2]> {
    let output = if output == [0, 0] { max_output } else { output };
    (fits(render, max_render) && fits(output, max_output) && fits(render, output)).then_some(output)
}

fn minimum_extent(index: usize, flags: u32, render: [u32; 2], output: [u32; 2]) -> [u32; 2] {
    match index {
        6 => output,
        2 if flags & DISPLAY_RESOLUTION_MOTION_VECTORS != 0 => output,
        3 => [1, 1], // Optional exposure texture.
        _ => render,
    }
}

macro_rules! once {
    ($level:ident, $($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::$level!($($arg)*));
    }};
}

/// Serves the title's FidelityFX calls with the PC upscaler in `dir`
/// (`VICTORY_FIDELITYFX_DIR`) if it offers one for the host GPU; else the
/// title's own DLLs run. Call once, before the import registry is first
/// used: it holds this module only when serving.
pub fn use_host_dir(dir: Option<&Path>) {
    let Some(dir) = dir else {
        tracing::info!(
            "no FidelityFX folder (VICTORY_FIDELITYFX_DIR): a title's own FidelityFX DLLs run"
        );
        return;
    };

    match load(dir) {
        Ok((host, versions)) => {
            tracing::info!(folder = %dir.display(), versions, "FidelityFX: the host's upscaler serves the title's");
            let _ = HOST.set(host);
        }
        Err(why) => tracing::warn!(
            folder = %dir.display(),
            "FidelityFX: {why}; the title's own DLLs run"
        ),
    }
}

/// Whether the host's FidelityFX serves the title's.
pub(crate) fn serving() -> bool {
    HOST.get().is_some()
}

fn load(dir: &Path) -> Result<(Host, String), String> {
    let mut failures = Vec::new();
    for dll in DLLS {
        let path = dir.join(dll);
        if path.is_file() {
            match load_dll(&path, dll) {
                Ok(provider) => return Ok(provider),
                Err(error) => failures.push(error),
            }
        }
    }
    if failures.is_empty() {
        return Err("no PC FidelityFX DLL in the configured folder".into());
    }
    Err(failures.join(", "))
}

fn load_dll(path: &Path, dll: &str) -> Result<(Host, String), String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32,
        LoadLibraryExW,
    };
    // The search flags below want a full path.
    let path = std::path::absolute(path).map_err(|e| e.to_string())?;
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    // Its own folder and System32 only, never the current folder.
    let search = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
    // SAFETY: a NUL-terminated path to the user's FidelityFX build.
    let module = unsafe { LoadLibraryExW(wide.as_ptr(), null_mut(), search) };
    if module.is_null() {
        return Err(format!(
            "{dll} did not load ({})",
            std::io::Error::last_os_error()
        ));
    }
    let library = Library(module);
    let find = |name: &CStr| {
        // SAFETY: a loaded module and a NUL-terminated name.
        unsafe { GetProcAddress(module, name.as_ptr().cast()) }
            .map(|f| f as *const c_void)
            .ok_or_else(|| format!("{dll} has no {}", name.to_string_lossy()))
    };
    let device = victory_gpu::external::host_device().ok_or("there is no host GPU")?;
    // SAFETY: the API's exports, with its signatures.
    let host = unsafe {
        Host {
            create: std::mem::transmute::<*const c_void, Create>(find(c"ffxCreateContext")?),
            destroy: std::mem::transmute::<*const c_void, Destroy>(find(c"ffxDestroyContext")?),
            configure: std::mem::transmute::<*const c_void, Call>(find(c"ffxConfigure")?),
            query: std::mem::transmute::<*const c_void, Call>(find(c"ffxQuery")?),
            dispatch: std::mem::transmute::<*const c_void, Call>(find(c"ffxDispatch")?),
            _library: library,
            device,
        }
    };
    let versions = versions(&host)?;
    Ok((host, versions))
}

/// The upscaler versions the host DLL offers for the host GPU, by name; an
/// error if none.
fn versions(host: &Host) -> Result<String, String> {
    let mut count = 0u64;
    let mut query = GetVersions {
        header: Header {
            kind: GET_VERSIONS,
            next: null_mut(),
        },
        create_kind: CREATE_UPSCALE,
        device: host.device.as_raw(),
        count: &mut count,
        ids: null_mut(),
        names: null_mut(),
    };
    // SAFETY: a version query naming the host device.
    let code = unsafe { (host.query)(null_mut(), &mut query.header) };
    if code != OK || count == 0 {
        return Err(format!(
            "it offers no upscaler for the host GPU (code {code})"
        ));
    }
    let mut names = vec![null::<c_char>(); count as usize];
    query.names = names.as_mut_ptr();
    // SAFETY: as above, with room for `count` names.
    let code = unsafe { (host.query)(null_mut(), &mut query.header) };
    if code != OK {
        return Err(format!(
            "upscaler version names could not be queried (code {code})"
        ));
    }
    let versions = names
        .iter()
        .filter(|n| !n.is_null())
        // SAFETY: NUL-terminated names the DLL keeps.
        .map(|&n| unsafe { CStr::from_ptr(n) }.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(", ");
    Ok(versions)
}

/// Puts the host backend after the title's upscaler description, in place
/// of the console's backend that follows it (where the title puts it);
/// anywhere else the console's is left for the host DLL to ignore, ours
/// coming first.
fn link(upscale: &mut CreateUpscale, backend: &mut Backend) {
    let after = upscale.header.next;
    // SAFETY: the title's chain.
    backend.header.next = match unsafe { after.as_ref() } {
        Some(h) if h.kind == CONSOLE_BACKEND => h.next,
        _ => after,
    };
    backend.header.kind = HOST_BACKEND;
    upscale.header.next = &raw mut backend.header;
}

/// A FidelityFX resource state as the host resource state it means (as the
/// PC backend reads it); `None` for generic read and indirect arguments,
/// which a host texture cannot be in.
fn host_state(state: u32) -> Option<u32> {
    Some(match state {
        0x1 | 0x80 => 0,  // common, present
        0x2 => 0x8,       // unordered access
        0x4 => 0x40,      // compute read
        0x8 => 0x80,      // pixel read
        0xc => 0xc0,      // both
        0x10 => 0x800,    // copy source
        0x20 => 0x400,    // copy destination
        0x100 => 0x4,     // render target
        0x200 => 0x10,    // depth write
        _ => return None, // generic read, indirect arguments
    })
}

extern "C" fn ffxCreateContext(
    context: *mut *mut c_void,
    desc: *mut Header,
    memory: *const c_void,
) -> u32 {
    HOST.get().map_or(NO_PROVIDER, |host| {
        create_context(host, context, desc, memory)
    })
}

fn create_context(
    host: &Host,
    context: *mut *mut c_void,
    desc: *mut Header,
    memory: *const c_void,
) -> u32 {
    if context.is_null() || desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a chain of descriptions.
    let kind = unsafe { (*desc).kind };
    if kind != CREATE_UPSCALE {
        once!(
            warn,
            kind = format_args!("{kind:#x}"),
            "FidelityFX: only the upscaler runs on the host; another effect was refused"
        );
        return NO_PROVIDER;
    }
    // SAFETY: an upscaler description, by its type.
    let mut upscale = Box::new(unsafe { desc.cast::<CreateUpscale>().read() });
    let mut backend = Box::new(Backend {
        header: Header {
            kind: HOST_BACKEND,
            next: null_mut(),
        },
        device: host.device.as_raw(),
    });
    link(&mut upscale, &mut backend);
    // SAFETY: the PC chain is retained until destroy; the title owns the remaining links.
    let code = unsafe { (host.create)(context, &mut upscale.header, memory) };
    if code != OK {
        tracing::error!(code, render = ?upscale.max_render, output = ?upscale.max_output, flags = format_args!("{:#x}", upscale.flags), "FidelityFX: the host refused the title's upscaler");
        return code;
    }
    let mut version = ProviderVersion {
        header: Header {
            kind: GET_PROVIDER_VERSION,
            next: null_mut(),
        },
        id: 0,
        name: null(),
    };
    // SAFETY: a provider query on the context just made.
    unsafe { (host.query)(context, &mut version.header) };
    // SAFETY: a NUL-terminated name the DLL keeps, if any.
    let name = (!version.name.is_null())
        .then(|| unsafe { CStr::from_ptr(version.name) }.to_string_lossy());
    tracing::info!(
        provider = name.as_deref().unwrap_or("?"),
        id = format_args!("{:#x}", version.id),
        render = ?upscale.max_render,
        output = ?upscale.max_output,
        flags = format_args!("{:#x}", upscale.flags),
        "FidelityFX: upscaler context on the host"
    );
    // SAFETY: a successful SDK creation wrote an opaque context into this slot.
    let handle = unsafe { *context } as usize;
    CONTEXTS.lock().unwrap_or_else(|e| e.into_inner()).insert(
        handle,
        Arc::new(Context {
            live: Mutex::new(true),
            creation: CreateChain {
                description: upscale,
                _backend: backend,
            },
        }),
    );
    OK
}

extern "C" fn ffxDestroyContext(context: *mut *mut c_void, memory: *const c_void) -> u32 {
    HOST.get()
        .map_or(NO_PROVIDER, |host| destroy_context(host, context, memory))
}

fn destroy_context(host: &Host, context: *mut *mut c_void, memory: *const c_void) -> u32 {
    if context.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes its context. Capture its value before destroy clears it.
    let handle = unsafe { *context } as usize;
    let ctx = CONTEXTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&handle)
        .cloned();
    let Some(ctx) = ctx else { return PARAMETER };
    let mut live = ctx.live.lock().unwrap_or_else(|e| e.into_inner());
    if !*live {
        return PARAMETER;
    }
    // The guest releases a context after submitting its frames. The host
    // may still be executing their lists, even after guest Execute returns.
    victory_gpu::external::finish_submitted();
    // SAFETY: the context is still live and its creation chain remains owned here.
    let code = unsafe { (host.destroy)(context, memory) };
    if code == OK {
        *live = false;
        let mut contexts = CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        // A concurrent Create may already have reused the released SDK address.
        if contexts
            .get(&handle)
            .is_some_and(|current| Arc::ptr_eq(current, &ctx))
        {
            contexts.remove(&handle);
        }
    }
    code
}

/// Settings (key-value pairs, debug messages) name nothing console-specific.
extern "C" fn ffxConfigure(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    // SAFETY: the title passes its context and descriptions.
    HOST.get()
        .map_or(NO_PROVIDER, |h| unsafe { (h.configure)(context, desc) })
}

extern "C" fn ffxQuery(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    HOST.get()
        .map_or(NO_PROVIDER, |host| query_info(host, context, desc))
}

fn query_info(host: &Host, context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a description.
    let kind = unsafe { (*desc).kind };
    // Substitute the native device; the SDK returns values through the
    // title's original output pointers in both request structures.
    if kind == GET_VERSIONS {
        // SAFETY: this kind identifies a complete version query.
        let mut query = unsafe { *desc.cast::<GetVersions>() };
        query.device = host.device.as_raw();
        // SAFETY: the copied description has a native device and the original output pointers.
        return unsafe { (host.query)(context, &mut query.header) };
    }
    if kind == MEMORY_USAGE_WITH_DEVICE {
        // SAFETY: this kind identifies a complete memory usage query.
        let mut query = unsafe { *desc.cast::<MemoryUsageWithDevice>() };
        query.device = host.device.as_raw();
        // SAFETY: the copied description has a native device and the original output pointers.
        return unsafe { (host.query)(context, &mut query.header) };
    }
    // SAFETY: as above.
    unsafe { (host.query)(context, desc) }
}

extern "C" fn ffxDispatch(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    let Some(host) = HOST.get() else {
        return NO_PROVIDER;
    };
    if desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a description.
    let kind = unsafe { (*desc).kind };
    if kind != DISPATCH_UPSCALE {
        once!(
            warn,
            kind = format_args!("{kind:#x}"),
            "FidelityFX: only upscaling runs on the host; a dispatch was refused"
        );
        return UNKNOWN_TYPE;
    }
    // SAFETY: an upscaling, by its type.
    let mut dispatch = unsafe { desc.cast::<DispatchUpscale>().read() };
    if context.is_null() {
        once!(warn, "FidelityFX dispatch arrived without a context");
        return PARAMETER;
    }
    if let Some(index) = [0, 1, 2, 6]
        .into_iter()
        .find(|&i| dispatch.resources[i].resource.is_null())
    {
        once!(
            warn,
            index,
            "FidelityFX dispatch is missing a required texture"
        );
        return PARAMETER;
    }
    // SAFETY: the title passes the same opaque context slot as for creation.
    let handle = unsafe { *context } as usize;
    let ctx = CONTEXTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&handle)
        .cloned();
    let Some(ctx) = ctx else {
        once!(
            warn,
            "FidelityFX dispatch referenced an unregistered context"
        );
        return PARAMETER;
    };
    let live = ctx.live.lock().unwrap_or_else(|e| e.into_inner());
    if !*live {
        return PARAMETER;
    }
    let create = &ctx.creation.description;
    let Some(output_size) = dispatch_output(
        dispatch.render_size,
        dispatch.upscale_size,
        create.max_render,
        create.max_output,
    ) else {
        once!(warn, render = ?dispatch.render_size, output = ?dispatch.upscale_size,
            max_render = ?create.max_render, max_output = ?create.max_output,
            "FidelityFX rejected dispatch dimensions outside the context's allocations");
        return PARAMETER;
    };
    let list = dispatch.list.cast::<Object>().cast_const();
    let output = dispatch.resources[6].resource.cast::<Object>();
    // Alive for the call (the list holds them too).
    let mut textures = Vec::new();
    for (index, r) in dispatch.resources.iter_mut().enumerate() {
        if r.resource.is_null() {
            continue;
        }
        if index < 6
            && victory_gpu::external::prepare_input(list, r.resource.cast::<Object>()).is_none()
        {
            once!(
                warn,
                index,
                "FidelityFX input could not be prepared on the title command list"
            );
            return PARAMETER;
        }
        let wanted = host_state(r.state);
        let texture = wanted.and_then(|w| {
            victory_gpu::external::host_texture(list, r.resource.cast::<Object>(), w)
                .filter(|&(_, state)| state == w)
        });
        let Some((texture, _)) = texture else {
            once!(
                warn,
                index,
                state = r.state,
                "FidelityFX: a resource has no host texture in its state; the upscaling did not run"
            );
            return PARAMETER;
        };
        // SAFETY: a live native texture.
        let desc = unsafe { texture.GetDesc() };
        let needed = minimum_extent(index, create.flags, dispatch.render_size, output_size);
        if desc.Width < u64::from(needed[0]) || desc.Height < needed[1] {
            once!(
                warn,
                index,
                ?needed,
                width = desc.Width,
                height = desc.Height,
                "FidelityFX resource cannot cover the active dispatch rectangle"
            );
            return PARAMETER;
        }
        r.resource = texture.as_raw();
        textures.push(texture);
    }
    once!(info, formats = ?textures.iter().map(|t| {
        // SAFETY: a live texture.
        unsafe { t.GetDesc() }.Format.0
    }).collect::<Vec<_>>(), "FidelityFX: first upscaling, host formats of its textures");
    let Some(code) = victory_gpu::external::record(list, |host_list| {
        dispatch.list = host_list.as_raw();
        // SAFETY: the PC dispatch on host objects; the context is the host's.
        unsafe { (host.dispatch)(context, &raw mut dispatch.header) }
    }) else {
        once!(
            warn,
            "FidelityFX could not record to the title command list"
        );
        return PARAMETER;
    };
    if code == OK {
        victory_gpu::external::output_written(list, output);
    } else {
        once!(
            warn,
            code,
            "FidelityFX provider rejected the title dispatch"
        );
    }
    code
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "fidelityfx",
        dlls: &["amd_fidelityfx_loader_xbox.dll"],
        exports: vec![
            hle_export!(ffxConfigure),
            hle_export!(ffxCreateContext),
            hle_export!(ffxDestroyContext),
            hle_export!(ffxDispatch),
            hle_export!(ffxQuery),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_keeps_its_creation_chain_and_receives_the_native_query_device() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use windows_sys::Win32::System::LibraryLoader::LoadLibraryW;

        static DEVICE: AtomicUsize = AtomicUsize::new(0);

        unsafe extern "C" fn create(
            context: *mut *mut c_void,
            desc: *mut Header,
            _: *const c_void,
        ) -> u32 {
            // SAFETY: called with a creation description and output slot by the bridge.
            unsafe { *context = desc.cast() };
            OK
        }

        unsafe extern "C" fn destroy(context: *mut *mut c_void, _: *const c_void) -> u32 {
            // SAFETY: the mock's handle is the retained creation description.
            let desc = unsafe { &*(*context).cast::<CreateUpscale>() };
            // SAFETY: creation links the retained native backend immediately after the descriptor.
            let backend = unsafe { &*desc.header.next.cast::<Backend>() };
            let device = DEVICE.load(Ordering::Relaxed);
            if desc.max_render != [640, 360]
                || backend.header.kind != HOST_BACKEND
                || backend.device as usize != device
            {
                return PARAMETER;
            }
            // SAFETY: the output slot is still live.
            unsafe { *context = null_mut() };
            OK
        }

        unsafe extern "C" fn query(_: *mut *mut c_void, desc: *mut Header) -> u32 {
            let device = DEVICE.load(Ordering::Relaxed);
            // SAFETY: each query's kind determines its descriptor type.
            unsafe {
                match (*desc).kind {
                    GET_VERSIONS => {
                        let q = &*desc.cast::<GetVersions>();
                        if q.device as usize != device {
                            return PARAMETER;
                        }
                        *q.count = 1;
                    }
                    MEMORY_USAGE_WITH_DEVICE => {
                        let q = &*desc.cast::<MemoryUsageWithDevice>();
                        if q.device as usize != device {
                            return PARAMETER;
                        }
                        *q.usage.cast::<u64>() = 4096;
                    }
                    _ => {}
                }
            }
            OK
        }

        let Some(device) = victory_gpu::external::host_device() else {
            eprintln!("no host GPU; skipped");
            return;
        };
        let name: Vec<_> = "kernel32.dll".encode_utf16().chain([0]).collect();
        // SAFETY: a terminated module name; Library owns the loaded reference.
        let module = unsafe { LoadLibraryW(name.as_ptr()) };
        assert!(!module.is_null());
        DEVICE.store(device.as_raw() as usize, Ordering::Relaxed);
        let host = Host {
            _library: Library(module),
            create,
            destroy,
            configure: query,
            query,
            dispatch: query,
            device,
        };

        let guest_device = std::ptr::dangling_mut::<c_void>();
        let mut device = Backend {
            header: header(CONSOLE_BACKEND, null_mut()),
            device: guest_device,
        };
        let mut desc = CreateUpscale {
            header: header(CREATE_UPSCALE, &mut device.header),
            flags: 0,
            max_render: [640, 360],
            max_output: [1280, 720],
            message: 0,
        };
        let mut context = null_mut();
        assert_eq!(
            create_context(&host, &mut context, &mut desc.header, null()),
            OK
        );
        let handle = context as usize;
        desc.max_render = [1, 1];
        assert_ne!(context, (&raw mut desc).cast());

        let mut count = 0;
        let mut versions = GetVersions {
            header: header(GET_VERSIONS, null_mut()),
            create_kind: CREATE_UPSCALE,
            device: guest_device,
            count: &mut count,
            ids: null_mut(),
            names: null_mut(),
        };
        assert_eq!(query_info(&host, null_mut(), &mut versions.header), OK);
        assert_eq!(count, 1);
        assert_eq!(versions.device, guest_device);

        let mut bytes = 0u64;
        let mut memory = MemoryUsageWithDevice {
            header: header(MEMORY_USAGE_WITH_DEVICE, null_mut()),
            device: guest_device,
            max_render: [640, 360],
            max_output: [1280, 720],
            flags: 0,
            usage: (&raw mut bytes).cast(),
        };
        assert_eq!(query_info(&host, null_mut(), &mut memory.header), OK);
        assert_eq!(bytes, 4096);
        assert_eq!(memory.device, guest_device);
        assert_eq!(destroy_context(&host, &mut context, null()), OK);
        assert!(context.is_null());
        assert!(!CONTEXTS.lock().unwrap().contains_key(&handle));
    }

    #[test]
    fn validates_logical_rectangles_against_the_current_context() {
        for display in [[1920, 1080], [2560, 1440], [3840, 2160]] {
            for divisor in [1.0_f32, 1.5, 1.7, 2.0, 3.0] {
                let render = display.map(|v| (v as f32 / divisor) as u32);
                assert_eq!(
                    dispatch_output(render, display, render, display),
                    Some(display)
                );
                assert_eq!(
                    dispatch_output(render, [0, 0], render, display),
                    Some(display)
                );
                assert_eq!(minimum_extent(2, 0x29, render, display), render);
                assert_eq!(minimum_extent(2, 0x2b, render, display), display);
                assert_eq!(minimum_extent(6, 0x29, render, display), display);
                assert_eq!(minimum_extent(3, 0x29, render, display), [1, 1]);
            }
        }
        assert_eq!(
            dispatch_output([0, 720], [0, 0], [1280, 720], [2560, 1440]),
            None
        );
        assert_eq!(
            dispatch_output([1281, 720], [0, 0], [1280, 720], [2560, 1440]),
            None
        );
        assert_eq!(
            dispatch_output([1280, 720], [0, 1440], [1280, 720], [2560, 1440]),
            None
        );
        assert_eq!(
            dispatch_output([1280, 720], [3000, 1440], [1280, 720], [2560, 1440]),
            None
        );
        assert_eq!(
            dispatch_output([1280, 720], [960, 540], [1280, 720], [2560, 1440]),
            None
        );
    }

    fn header(kind: u64, next: *mut Header) -> Header {
        Header { kind, next }
    }

    /// The title's chain: upscaler, console backend, version.
    #[test]
    fn the_console_backend_becomes_the_hosts() {
        let mut version = header(0x1_000b, null_mut());
        let mut console = Backend {
            header: header(CONSOLE_BACKEND, &raw mut version),
            device: 0x1234 as *mut c_void,
        };
        let mut upscale = CreateUpscale {
            header: header(CREATE_UPSCALE, &raw mut console.header),
            flags: 0x29,
            max_render: [1280, 720],
            max_output: [1920, 1080],
            message: 0,
        };
        let mut backend = Backend {
            header: header(0, null_mut()),
            device: 0x5678 as *mut c_void,
        };
        link(&mut upscale, &mut backend);
        assert_eq!(upscale.header.next, &raw mut backend.header);
        assert_eq!(backend.header.kind, HOST_BACKEND);
        assert_eq!(backend.header.next, &raw mut version);
        assert_eq!(backend.device, 0x5678 as *mut c_void);
        // Without a console backend right after it, ours goes in front.
        let mut upscale = CreateUpscale {
            header: header(CREATE_UPSCALE, &raw mut version),
            ..upscale
        };
        link(&mut upscale, &mut backend);
        assert_eq!(backend.header.next, &raw mut version);
    }

    /// Without the PC build the title's own DLL serves its imports.
    #[test]
    fn without_the_dll_the_titles_own_runs() {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            load(dir.path())
                .err()
                .unwrap()
                .contains("no PC FidelityFX DLL")
        );
        assert!(!serving());
        let symbol = victory_pe::ImportSymbol::Name {
            hint: 0,
            name: "ffxCreateContext".into(),
        };
        let resolver = crate::HleResolver::new(crate::HostPolicy::none());
        assert_eq!(
            resolver.classify("amd_fidelityfx_loader_xbox.dll", &symbol),
            None
        );
    }

    #[test]
    fn resource_states_read_as_the_pc_backend_reads_them() {
        assert_eq!(host_state(0xc), Some(0xc0));
        assert_eq!(host_state(0x2), Some(0x8));
        assert_eq!(host_state(0x200), Some(0x10));
        assert_eq!(host_state(0x14), None);
    }
}
