use super::{Lower, V};

impl Lower<'_> {
    /// Reads up to four consecutive dwords of a single console load.
    pub(super) fn raw_dword_group(&mut self, handle: V, address: V, count: u32) -> V {
        let undef = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8((1u8 << count) - 1);
        let alignment = self.s.m.i32(4);
        self.raw_buffer_load(self.cur, &[handle, address, undef, mask, alignment])
    }
}
