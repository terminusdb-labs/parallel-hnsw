use crate::{AbstractVector, Comparator, Hnsw, VectorId};
use std::sync::Arc;

trait Quantizer<const SIZE: usize, const SUBDIMENSION: usize> {
    fn quantize(&self, vec: &[f32; SIZE]) -> [u16; SUBDIMENSION];
    fn reconstruct(&self, qvec: &[u16; SUBDIMENSION]) -> [f32; SIZE];
}

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

pub struct QuantizedHnsw<
    const SIZE: usize,
    const SUBDIMENSION: usize,
    CentroidComparator: Comparator<T = [f32]> + 'static,
    QuantizedComparator: Comparator<T = [u16]> + 'static,
    FullComparator: Comparator<T = [f32; SIZE]> + 'static,
> {
    quantizer: HnswQuantizer<SIZE, SUBDIMENSION, CentroidComparator>,
    hnsw: Hnsw<QuantizedComparator>,
    comparator: FullComparator,
}

impl<
        const SIZE: usize,
        const SUBDIMENSION: usize,
        CentroidComparator: Comparator<T = [f32]> + 'static,
        QuantizedComparator: Comparator<T = [u16]> + 'static,
        FullComparator: Comparator<T = [f32; SIZE]> + 'static,
    > QuantizedHnsw<SIZE, SUBDIMENSION, CentroidComparator, QuantizedComparator, FullComparator>
{
    pub fn new() -> Self {
        todo!()
    }

    pub fn search(
        &self,
        v: AbstractVector<[f32; SIZE]>,
        number_of_candidates: usize,
        probe_depth: usize,
    ) -> Vec<(VectorId, f32)> {
        let raw_v = self.comparator.lookup_abstract(v);
        let quantized = self.quantizer.quantize(raw_v);
        let result = self.hnsw.search(
            AbstractVector::Unstored(&quantized),
            number_of_candidates,
            probe_depth,
        );
        // TODO reorder
        result
    }
}
