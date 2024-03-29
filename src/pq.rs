use crate::{AbstractVector, Comparator, Hnsw, VectorId};
use linfa::traits::Fit;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use ndarray::Array;
use rand::{rngs::StdRng, SeedableRng};

trait Quantizer<const SIZE: usize, const QUANTIZED_SIZE: usize> {
    fn quantize(&self, vec: &[f32; SIZE]) -> [u16; QUANTIZED_SIZE];
    fn reconstruct(&self, qvec: &[u16; QUANTIZED_SIZE]) -> [f32; SIZE];
}

pub struct HnswQuantizer<
    const SIZE: usize,
    const CENTROID_SIZE: usize,
    const QUANTIZED_SIZE: usize,
    C: Comparator<T = [f32; CENTROID_SIZE]>,
> {
    hnsw: Hnsw<C>,
}

impl<
        const SIZE: usize,
        const CENTROID_SIZE: usize,
        const QUANTIZED_SIZE: usize,
        C: 'static + Comparator<T = [f32; CENTROID_SIZE]>,
    > Quantizer<SIZE, QUANTIZED_SIZE> for HnswQuantizer<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, C>
{
    fn quantize(&self, vec: &[f32; SIZE]) -> [u16; QUANTIZED_SIZE] {
        let mut result = [0; QUANTIZED_SIZE];
        for (ix, v) in vec.chunks(CENTROID_SIZE).enumerate() {
            let v: &[f32; CENTROID_SIZE] = unsafe { &*(v.as_ptr() as *const [f32; CENTROID_SIZE]) };
            let distances = self.hnsw.search(AbstractVector::Unstored(v), 100, 2);
            let quant = distances[0].0 .0 as u16; // TODO maybe debug assert
            result[ix] = quant;
        }
        result
    }

    fn reconstruct(&self, qvec: &[u16; QUANTIZED_SIZE]) -> [f32; SIZE] {
        let mut result = [0.0; SIZE];
        for (ix, i) in qvec.iter().enumerate() {
            let slice = &mut result[ix * CENTROID_SIZE..(ix + 1) * CENTROID_SIZE];
            let centroid = self.hnsw.comparator().lookup(VectorId(*i as usize));
            slice.copy_from_slice(&*centroid);
        }
        result
    }
}

pub struct QuantizedHnsw<
    const SIZE: usize,
    const CENTROID_SIZE: usize,
    const QUANTIZED_SIZE: usize,
    CentroidComparator: Comparator<T = [f32; CENTROID_SIZE]> + 'static + VectorStore<T = [f32; CENTROID_SIZE]>,
    QuantizedComparator: Comparator<T = [u16; QUANTIZED_SIZE]> + 'static,
    FullComparator: Comparator<T = [f32; SIZE]> + VectorSelector<T = [f32; SIZE]> + 'static,
> {
    quantizer: HnswQuantizer<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, CentroidComparator>,
    hnsw: Hnsw<QuantizedComparator>,
    comparator: FullComparator,
}

pub trait VectorSelector {
    type T;
    fn selection(&self, size: usize) -> Vec<Self::T>;
    fn vector_chunks(&self) -> impl Iterator<Item = Vec<Self::T>>;
}

pub trait VectorStore {
    type T;
    fn store(&self, i: Box<dyn Iterator<Item = Self::T>>) -> Vec<VectorId>;
}

impl<
        const SIZE: usize,
        const CENTROID_SIZE: usize,
        const QUANTIZED_SIZE: usize,
        CentroidComparator: Comparator<T = [f32; CENTROID_SIZE]> + VectorStore<T = [f32; CENTROID_SIZE]> + 'static,
        QuantizedComparator: Comparator<T = [u16; QUANTIZED_SIZE]> + VectorStore<T = [u16; QUANTIZED_SIZE]> + 'static,
        FullComparator: Comparator<T = [f32; SIZE]> + VectorSelector<T = [f32; SIZE]> + 'static,
    >
    QuantizedHnsw<
        SIZE,
        CENTROID_SIZE,
        QUANTIZED_SIZE,
        CentroidComparator,
        QuantizedComparator,
        FullComparator,
    >
{
    pub fn new(
        selection_size: usize,
        centroid_comparator: CentroidComparator,
        quantized_comparator: QuantizedComparator,
        comparator: FullComparator,
    ) -> Self {
        let vector_selection = comparator.selection(selection_size);
        // Linfa
        let data: Vec<f32> = vector_selection
            .into_iter()
            .flat_map(|v| v.into_iter())
            .collect();
        let sub_length = selection_size * SIZE / CENTROID_SIZE;
        let sub_arrays = Array::from_shape_vec((CENTROID_SIZE, sub_length), data).unwrap();
        let observations = DatasetBase::from(sub_arrays.clone());
        // TODO review this number
        let number_of_clusters = selection_size;
        let prng = StdRng::seed_from_u64(42);
        let model = KMeans::params_with_rng(number_of_clusters, prng.clone())
            .tolerance(1e-2)
            .fit(&observations)
            .expect("KMeans fitted");
        let centroid_flat: Vec<f32> = model.centroids().clone().into_raw_vec();
        let centroids: Vec<[f32; CENTROID_SIZE]> = centroid_flat
            .chunks(CENTROID_SIZE)
            .map(|v| {
                let mut array = [0.0; CENTROID_SIZE];
                array.copy_from_slice(v);
                array
            })
            .collect();
        //
        let vector_ids = centroid_comparator.store(Box::new(centroids.into_iter()));
        let centroid_m = 24;
        let centroid_m0 = 48;
        let centroid_order = 12;
        let centroid_hnsw: Hnsw<CentroidComparator> = Hnsw::generate(
            centroid_comparator,
            vector_ids,
            centroid_m,
            centroid_m0,
            centroid_order,
        );
        let centroid_quantizer: HnswQuantizer<
            SIZE,
            CENTROID_SIZE,
            QUANTIZED_SIZE,
            CentroidComparator,
        > = HnswQuantizer {
            hnsw: centroid_hnsw,
        };
        let mut vids: Vec<VectorId> = Vec::new();
        for chunk in comparator.vector_chunks() {
            let mut quantized = Vec::new();
            for v in chunk {
                quantized.push(centroid_quantizer.quantize(&v));
            }
            vids.extend(quantized_comparator.store(Box::new(quantized.into_iter())));
        }
        let m = 24;
        let m0 = 48;
        let order = 12;
        let hnsw: Hnsw<QuantizedComparator> =
            Hnsw::generate(quantized_comparator, vids, m, m0, order);
        Self {
            quantizer: centroid_quantizer,
            hnsw,
            comparator,
        }
    }

    pub fn search(
        &self,
        v: AbstractVector<[f32; SIZE]>,
        number_of_candidates: usize,
        probe_depth: usize,
    ) -> Vec<(VectorId, f32)> {
        let raw_v = self.comparator.lookup_abstract(v);
        let quantized = self.quantizer.quantize(&raw_v);
        let result = self.hnsw.search(
            AbstractVector::Unstored(&quantized),
            number_of_candidates,
            probe_depth,
        );
        // TODO reorder
        result
    }
}

mod tests {
    // assumes normalized vectors
    fn cosine32(v1: &[f32; 32], v2: &[f32; 32]) -> f32 {
        (1.0 - v1
            .iter()
            .zip(v2.iter())
            .map(|(f1, f2)| f1 * f2)
            .sum::<f32>())
            / 2.0
    }

    use std::{
        ops::Deref,
        sync::{Arc, RwLock, RwLockReadGuard},
    };

    use crate::{Comparator, VectorId};

    use super::VectorStore;

    struct ReadLockedVec<'a> {
        lock: RwLockReadGuard<'a, Vec<[f32; 32]>>,
        id: VectorId,
    }

    impl<'a> Deref for ReadLockedVec<'a> {
        type Target = [f32; 32];

        fn deref(&self) -> &Self::Target {
            &self.lock[self.id.0]
        }
    }

    #[derive(Clone)]
    struct CentroidComparator32 {
        data: Arc<RwLock<Vec<[f32; 32]>>>,
    }
    impl Comparator for CentroidComparator32 {
        type T = [f32; 32];
        type Borrowable<'a> = ReadLockedVec<'a>;
        fn lookup(&self, v: crate::VectorId) -> Self::Borrowable<'_> {
            ReadLockedVec {
                lock: self.data.read().unwrap(),
                id: v,
            }
        }

        fn compare_raw(&self, v1: &Self::T, v2: &Self::T) -> f32 {
            cosine32(v1, v2)
        }
    }

    impl VectorStore for CentroidComparator32 {
        type T = <CentroidComparator32 as Comparator>::T;

        fn store(&self, i: Box<dyn Iterator<Item = Self::T>>) -> Vec<VectorId> {
            let mut data = self.data.write().unwrap();
            let vid = data.len();
            let mut vectors: Vec<VectorId> = Vec::new();
            data.extend(i.enumerate().map(|(i, v)| {
                vectors.push(VectorId(vid + i));
                v
            }));
            vectors
        }
    }

    #[test]
    fn test_pq() {
        todo!();
    }
}
