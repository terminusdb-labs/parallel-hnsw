use std::fmt::Debug;

#[derive(PartialEq, Eq, Debug, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct VectorId(pub usize);
#[derive(PartialEq, Eq, Debug, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct NodeId(pub usize);

impl VectorId {
    pub const MAX: Self = Self(!0);
}
impl NodeId {
    pub const MAX: Self = Self(!0);
}

pub trait EmptyValue {
    fn is_empty(&self) -> bool;
    fn empty() -> Self;
}

impl EmptyValue for NodeId {
    fn is_empty(&self) -> bool {
        self.0 == !0
    }

    fn empty() -> Self {
        Self::MAX
    }
}

impl EmptyValue for VectorId {
    fn is_empty(&self) -> bool {
        self.0 == !0
    }

    fn empty() -> Self {
        Self::MAX
    }
}

pub enum AbstractVector<'a, T> {
    Stored(VectorId),
    Unstored(&'a T),
}

impl<'a, T> Debug for AbstractVector<'a, T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stored(arg0) => f.debug_tuple("Stored").field(arg0).finish(),
            Self::Unstored(arg0) => f.debug_tuple("Unstored").field(arg0).finish(),
        }
    }
}

impl<'a, T> Clone for AbstractVector<'a, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Stored(arg0) => Self::Stored(*arg0),
            Self::Unstored(arg0) => Self::Unstored(arg0),
        }
    }
}

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct OrderedFloat(pub f32);

impl Eq for OrderedFloat {}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
