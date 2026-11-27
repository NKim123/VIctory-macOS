//! Host work that a host library records into a title's command list in
//! place of a console one (the host's FidelityFX upscaler, say): the host
//! device, a title texture's host copy in the state the library is told,
//! and the host list itself.

use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12GraphicsCommandList, ID3D12Resource,
};

use crate::object::Object;

/// The host D3D12 device, created on first use; `None` without a host GPU.
pub fn host_device() -> Option<ID3D12Device> {
    crate::host::host().map(|h| h.device.clone())
}

/// A title texture's host copy, moved to the host resource `state` in the
/// title list `list`, and the state it is in: `state` without the bits the
/// host copy cannot take (see `host::legal_state`). What the list records
/// next finds it there and must leave it there. The list holds the texture
/// until it is reset. `None` for a buffer, or without a host list.
pub fn host_texture(
    list: *const Object,
    resource: *const Object,
    state: u32,
) -> Option<(ID3D12Resource, u32)> {
    let host = crate::list::host_list(list)?;
    let texture = crate::resource::host_resource(resource)?.clone();
    crate::list::ensure_state(host, resource, state);
    let state = crate::host::legal_state(&texture, state);
    Some((texture, state))
}

/// Sync a title input's host copy from its GPU alias or title memory.
/// `host_texture` subsequently routes depth-twin reads during its transition.
pub fn prepare_input(list: *const Object, resource: *const Object) -> Option<()> {
    let list = crate::list::host_list(list)?;
    let copy = crate::resource::host_copy(resource)?;
    let aliased = crate::resource::alias_source(resource)
        .is_some_and(|source| crate::texture::alias_copy(list, source.get(), resource));
    if !aliased && copy.title_written() {
        crate::texture::upload(list, resource, &copy.texture);
    }
    Some(())
}

/// Mark the provider's output as host-authored when its list is submitted.
pub fn output_written(list: *const Object, resource: *const Object) {
    if let Some(list) = crate::list::host_list(list) {
        crate::texture::host_writes(list, resource);
    }
}

/// Wait for the native submissions being pumped by the title queues.
/// This backend completes the host fence synchronously inside the queue pump.
pub fn finish_submitted() {
    crate::queue::pump_held(crate::queue::Why::Fence);
}

/// Runs `f` with the host list behind the title list `list` (open for
/// recording); `None` without one. What `f` binds (descriptor heaps, root
/// signatures, pipeline) is replaced before the list's next draw or
/// dispatch.
pub fn record<R>(
    list: *const Object,
    f: impl FnOnce(&ID3D12GraphicsCommandList) -> R,
) -> Option<R> {
    let mut out = None;
    crate::list::with_state(list, |host, state| {
        out = Some(f(&host.list));
        state.host_state_lost();
    });
    out
}
