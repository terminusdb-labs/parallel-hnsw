use crate::{Comparator, Hnsw};
use std::sync::Arc;

trait Quantizer<const SUBDIMENSION: usize> {
    type T;
    fn quantize(&self, vec: &Self::T) -> &[f32; SUBDIMENSION];
    fn reconstruct(&self, qvec: &[u16; SUBDIMENSION]) -> Self::T;
}

pub struct QuantizedHnsw<CentroidComparator: Comparator, QuantizedComparator: Comparator> {
    centroid_size: usize,
    centroids: Arc<Vec<CentroidComparator::T>>,
    data: Arc<Vec<QuantizedComparator::T>>,
    hnsw: Arc<Hnsw<CentroidComparator>>,
}

pub struct QuantizedComparator {
    comparator: Comparator,
}

const THIRTY_TWO_QUANTIZER: usize = 32;
impl Quantizer<THIRTY_TWO_QUANTIZER> for QuantizedHnsw<Comparator> {
    type T = Vec<f32>;

    fn quantize(&self, vec: &Self::T) -> &[f32; THIRTY_TWO_QUANTIZER] {
        let len = vec.len();
        let parts = len / THIRTY_TWO_QUANTIZER;
        let mut vec: Vec<usize> = Vec::with_capacity(parts);
        for v in vin.chunks(parts) {
            let distances = self
                .hnsw
                .search(AbstractVector::Unstored(&v.to_vec()), 100, 2);
            vec.push(distances[0].0 .0)
        }
        vec
    }

    fn reconstruct(&self, qvec: &[u16; THIRTY_TWO_QUANTIZER]) -> Self::T {
        let size = self.centroid_size * vin.len();
        let mut v = Vec::with_capacity(size);
        for i in qvec {
            v.extend(self.centroids[*i].iter())
        }
        v
    }
}
