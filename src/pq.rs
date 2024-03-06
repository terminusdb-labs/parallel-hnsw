use std::path::PathBuf;

use crate::{AbstractVector, Comparator, Hnsw, Serializable, VectorId};
use linfa::traits::Fit;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use ndarray::{Array, Array1, Array2};
use rand::{rngs::StdRng, SeedableRng};
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

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

impl<
        const SIZE: usize,
        const CENTROID_SIZE: usize,
        const QUANTIZED_SIZE: usize,
        ComparatorParams,
        C: 'static + Comparator<T = [f32; CENTROID_SIZE]> + Serializable<Params = ComparatorParams>,
    > Serializable for HnswQuantizer<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, C>
{
    type Params = ComparatorParams;

    fn serialize<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), crate::SerializationError> {
        self.hnsw.serialize(path)
    }

    fn deserialize<P: AsRef<std::path::Path>>(
        path: P,
        params: Self::Params,
    ) -> Result<Self, crate::SerializationError> {
        let hnsw = Hnsw::deserialize(path, params)?.unwrap();
        Ok(Self { hnsw })
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
        let sub_arrays = Array::from_shape_vec((sub_length, CENTROID_SIZE), data).unwrap();
        eprintln!("sub_arrays: {sub_arrays:?}");
        let observations = DatasetBase::from(sub_arrays);
        // TODO review this number
        let number_of_clusters = selection_size;
        let prng = StdRng::seed_from_u64(42);
        eprintln!("Running kmeans");
        let model = KMeans::params_with_rng(number_of_clusters, prng.clone())
            .tolerance(1e-2)
            .fit(&observations)
            .expect("KMeans fitted");
        let centroid_array: Array2<f32> = model.centroids().clone();
        let centroid_flat: Vec<f32> = centroid_array
            .into_shape(number_of_clusters * CENTROID_SIZE)
            .unwrap()
            .to_vec();
        let centroids: Vec<[f32; CENTROID_SIZE]> = centroid_flat
            .chunks(CENTROID_SIZE)
            .map(|v| {
                let mut array = [0.0; CENTROID_SIZE];
                array.copy_from_slice(v);
                array
            })
            .collect();
        //
        eprintln!("Number of centroids: {}", centroids.len());

        let vector_ids = centroid_comparator.store(Box::new(centroids.into_iter()));
        let centroid_m = 24;
        let centroid_m0 = 48;
        let centroid_order = 12;
        let mut centroid_hnsw: Hnsw<CentroidComparator> = Hnsw::generate(
            centroid_comparator,
            vector_ids,
            centroid_m,
            centroid_m0,
            centroid_order,
        );
        //centroid_hnsw.improve_index();
        centroid_hnsw.improve_neighbors(0.01);
        let centroid_quantizer: HnswQuantizer<
            SIZE,
            CENTROID_SIZE,
            QUANTIZED_SIZE,
            CentroidComparator,
        > = HnswQuantizer {
            hnsw: centroid_hnsw,
        };
        let mut vids: Vec<VectorId> = Vec::new();
        eprintln!("quantizing");
        for chunk in comparator.vector_chunks() {
            let quantized: Vec<_> = chunk
                .into_par_iter()
                .map(|v| centroid_quantizer.quantize(&v))
                .collect();

            vids.extend(quantized_comparator.store(Box::new(quantized.into_iter())));
        }
        let m = 24;
        let m0 = 48;
        let order = 12;
        eprintln!("generating");
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

impl<
        const SIZE: usize,
        const CENTROID_SIZE: usize,
        const QUANTIZED_SIZE: usize,
        ComparatorParams,
        CentroidComparator: Comparator<T = [f32; CENTROID_SIZE]>
            + VectorStore<T = [f32; CENTROID_SIZE]>
            + Serializable<Params = ()>
            + 'static,
        QuantizedComparator: Comparator<T = [u16; QUANTIZED_SIZE]>
            + VectorStore<T = [u16; QUANTIZED_SIZE]>
            + Serializable<Params = ()>
            + 'static,
        FullComparator: Comparator<T = [f32; SIZE]>
            + VectorSelector<T = [f32; SIZE]>
            + Serializable<Params = ComparatorParams>
            + 'static,
    > Serializable
    for QuantizedHnsw<
        SIZE,
        CENTROID_SIZE,
        QUANTIZED_SIZE,
        CentroidComparator,
        QuantizedComparator,
        FullComparator,
    >
{
    type Params = ComparatorParams;

    fn serialize<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), crate::SerializationError> {
        let path_buf: PathBuf = path.as_ref().into();

        let quantizer_path = path_buf.join("quantizer");
        eprintln!("serializing quantizer");
        self.quantizer.serialize(quantizer_path)?;

        let hnsw_path = path_buf.join("hnsw");
        eprintln!("serializing quantizer");
        self.hnsw.serialize(hnsw_path)?;

        let comparator_path = path_buf.join("comparator");
        eprintln!("serializing quantizer");
        self.comparator.serialize(comparator_path)?;

        Ok(())
    }

    fn deserialize<P: AsRef<std::path::Path>>(
        path: P,
        params: Self::Params,
    ) -> Result<Self, crate::SerializationError> {
        let path_buf: PathBuf = path.as_ref().into();

        let quantizer_path = path_buf.join("quantizer");
        let quantizer = HnswQuantizer::deserialize(quantizer_path, ())?;

        let hnsw_path = path_buf.join("hnsw");
        let hnsw = Hnsw::deserialize(hnsw_path, ())?.unwrap();

        let comparator_path = path_buf.join("comparator");
        let comparator = FullComparator::deserialize(comparator_path, params)?;

        Ok(Self {
            quantizer,
            hnsw,
            comparator,
        })
    }
}

mod tests {
    fn clamp_01(f: f32) -> f32 {
        if f <= 0.0 {
            0.0
        } else if f >= 1.0 {
            1.0
        } else {
            f
        }
    }

    fn normalize_cosine_distance(f: f32) -> f32 {
        clamp_01((f - 1.0) / -2.0)
    }

    // assumes normalized vectors
    fn cosine32(v1: &[f32; 32], v2: &[f32; 32]) -> f32 {
        normalize_cosine_distance(
            v1.iter()
                .zip(v2.iter())
                .map(|(f1, f2)| f1 * f2)
                .sum::<f32>(),
        )
    }

    fn cosine1536(v1: &[f32; 1536], v2: &[f32; 1536]) -> f32 {
        normalize_cosine_distance(
            v1.iter()
                .zip(v2.iter())
                .map(|(f1, f2)| f1 * f2)
                .sum::<f32>(),
        )
    }

    use std::{
        ops::Deref,
        sync::{Arc, RwLock, RwLockReadGuard},
    };

    use ndarray::Array2;
    use rand::{rngs::StdRng, SeedableRng};

    use crate::{
        bigvec::random_normed_vec, pq::QuantizedHnsw, AbstractVector, Comparator, VectorId,
    };
    use rayon::prelude::*;

    use super::{VectorSelector, VectorStore};

    struct ReadLockedVec<'a, T> {
        lock: RwLockReadGuard<'a, Vec<T>>,
        id: VectorId,
    }

    impl<'a, T> Deref for ReadLockedVec<'a, T> {
        type Target = T;

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
        type Borrowable<'a> = ReadLockedVec<'a, Self::T>;
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

    #[derive(Clone)]
    struct QuantizedComparator32 {
        cc: CentroidComparator32,
        data: Arc<RwLock<Vec<[u16; 48]>>>,
    }

    impl Comparator for QuantizedComparator32 {
        type T = [u16; 48];
        type Borrowable<'a> = ReadLockedVec<'a, Self::T>;
        fn lookup(&self, v: crate::VectorId) -> Self::Borrowable<'_> {
            ReadLockedVec {
                lock: self.data.read().unwrap(),
                id: v,
            }
        }

        fn compare_raw(&self, v1: &Self::T, v2: &Self::T) -> f32 {
            let v_reconstruct1: Vec<f32> = v1
                .iter()
                .flat_map(|i| self.cc.lookup(VectorId(*i as usize)).into_iter())
                .collect();
            let v_reconstruct2: Vec<f32> = v2
                .iter()
                .flat_map(|i| self.cc.lookup(VectorId(*i as usize)).into_iter())
                .collect();
            let mut ar1 = [0.0_f32; 1536];
            let mut ar2 = [0.0_f32; 1536];
            ar1.copy_from_slice(&v_reconstruct1);
            ar2.copy_from_slice(&v_reconstruct2);
            cosine1536(&ar1, &ar2)
        }
    }

    impl VectorStore for QuantizedComparator32 {
        type T = <QuantizedComparator32 as Comparator>::T;

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

    #[derive(Clone)]
    struct AIComparator {
        data: Arc<RwLock<Vec<[f32; 1536]>>>,
    }

    impl Comparator for AIComparator {
        type T = [f32; 1536];
        type Borrowable<'a> = ReadLockedVec<'a, Self::T>;
        fn lookup(&self, v: crate::VectorId) -> Self::Borrowable<'_> {
            ReadLockedVec {
                lock: self.data.read().unwrap(),
                id: v,
            }
        }

        fn compare_raw(&self, _v1: &Self::T, _v2: &Self::T) -> f32 {
            todo!()
        }
    }

    impl VectorStore for AIComparator {
        type T = <AIComparator as Comparator>::T;

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

    impl VectorSelector for AIComparator {
        type T = <AIComparator as Comparator>::T;

        fn selection(&self, size: usize) -> Vec<Self::T> {
            self.data
                .read()
                .unwrap()
                .iter()
                .cloned()
                .take(size)
                .collect()
        }

        fn vector_chunks(&self) -> impl Iterator<Item = Vec<Self::T>> {
            vec![self.data.read().unwrap().clone()].into_iter()
        }
    }

    #[test]
    fn test_arrays() {
        let a = Array2::from_shape_vec((2, 3), vec![0, 1, 2, 3, 4, 5]).unwrap();
        let b = a.t();
        let x = a.clone().into_raw_vec();
        eprintln!("x: {x:?}");
        let y = b.into_owned().into_raw_vec();
        eprintln!("y: {y:?}");
        panic!();
    }

    #[test]
    fn test_pq() {
        let count = 100;
        let vecs: Vec<[f32; 1536]> = (0..count)
            .into_par_iter()
            .map(move |i| {
                let mut prng = StdRng::seed_from_u64(42_u64 + i as u64);
                let mut arr = [0.0_f32; 1536];
                let v = random_normed_vec(&mut prng, 1536);
                arr.copy_from_slice(&v);
                arr
            })
            .collect();
        let cc = CentroidComparator32 {
            data: Arc::new(RwLock::new(Vec::new())),
        };
        let qc = QuantizedComparator32 {
            cc: cc.clone(),
            data: Arc::new(RwLock::new(Vec::new())),
        };
        let fc = AIComparator {
            data: Arc::new(RwLock::new(vecs.clone())),
        };
        let hnsw: QuantizedHnsw<1536, 32, 48, _, _, _> = QuantizedHnsw::new(100, cc, qc, fc);
        let v = AbstractVector::Unstored(&vecs[0]);
        let res = hnsw.search(v, 10, 2);
        eprintln!("res: {res:?}");
        panic!();
    }
}
