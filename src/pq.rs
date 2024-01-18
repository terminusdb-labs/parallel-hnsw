use crate::{AbstractVector, Comparator, Hnsw};
use std::sync::Arc;

trait Quantizer<const SIZE: usize, const SUBDIMENSION: usize> {
    fn quantize(&self, vec: &[f32; SIZE]) -> [u16; SUBDIMENSION];
    fn reconstruct(&self, qvec: &[u16; SUBDIMENSION]) -> [f32; SIZE];
}

/*
pub struct QuantizedHnsw<CentroidComparator: Comparator, QuantizedComparator: Comparator> {
    centroid_size: usize,
    centroids: Arc<Vec<CentroidComparator::T>>,
    data: Arc<Vec<QuantizedComparator::T>>,
    hnsw: Arc<Hnsw<CentroidComparator>>,
}

pub struct QuantizedComparator {
    comparator: Comparator,
}
*/

pub struct HnswQuantizer<const SIZE: usize, const SUBDIMENSION: usize, C: Comparator<T = [f32]>> {
    centroids: Arc<Vec<[f32; SUBDIMENSION]>>,
    hnsw: Hnsw<C>,
}

impl<const SIZE: usize, const SUBDIMENSION: usize, C: 'static + Comparator<T = [f32]>>
    Quantizer<SIZE, SUBDIMENSION> for HnswQuantizer<SIZE, SUBDIMENSION, C>
{
    fn quantize(&self, vec: &[f32; SIZE]) -> [u16; SUBDIMENSION] {
        let mut result = [0; SUBDIMENSION];
        let parts = SIZE / SUBDIMENSION;
        debug_assert!(SIZE % SUBDIMENSION == 0);
        for (ix, v) in vec.chunks(parts).enumerate() {
            let distances = self.hnsw.search(AbstractVector::Unstored(v), 100, 2);
            let quant = distances[0].0 .0 as u16; // TODO maybe debug assert
            result[ix] = quant;
        }
        result
    }

    fn reconstruct(&self, qvec: &[u16; SUBDIMENSION]) -> [f32; SIZE] {
        let mut result = [0.0; SIZE];
        for (ix, i) in qvec.iter().enumerate() {
            let slice = &mut result[ix * SUBDIMENSION..(ix + 1) * SUBDIMENSION];
            let centroid: &[f32; SUBDIMENSION] = &self.centroids[*i as usize];
            slice.copy_from_slice(centroid);
        }
        result
    }
}

/*
impl Quantizer<32> for QuantizedHnsw<Quantized> {
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
*/
