use std::path::PathBuf;

use crate::{AbstractVector, Comparator, Hnsw, OrderedFloat, Serializable, VectorId};
use chrono::Utc;
use linfa::traits::Fit;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use ndarray::{Array, Array2};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

pub trait Quantizer<const SIZE: usize, const QUANTIZED_SIZE: usize> {
    fn quantize(&self, vec: &[f32; SIZE]) -> [u16; QUANTIZED_SIZE];
    fn reconstruct(&self, qvec: &[u16; QUANTIZED_SIZE]) -> [f32; SIZE];
}

pub trait PartialDistance {
    // A partial distance must be a sumable component
    fn partial_distance(&self, i: u16, j: u16) -> f32;
}

pub struct HnswQuantizer<
    const SIZE: usize,
    const CENTROID_SIZE: usize,
    const QUANTIZED_SIZE: usize,
    C,
> {
    hnsw: Hnsw<C>,
}

impl<
        const SIZE: usize,
        const CENTROID_SIZE: usize,
        const QUANTIZED_SIZE: usize,
        C: 'static + Comparator<T = [f32; CENTROID_SIZE]>,
    > HnswQuantizer<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, C>
{
    pub fn new(hnsw: Hnsw<C>) -> Self {
        Self { hnsw }
    }

    pub fn comparator(&self) -> &C {
        self.hnsw.comparator()
    }
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
        C: 'static + Serializable<Params = ComparatorParams> + Clone + Sync,
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
        let hnsw = Hnsw::deserialize(path, params)?;
        Ok(Self { hnsw })
    }
}

pub struct QuantizedHnsw<
    const SIZE: usize,
    const CENTROID_SIZE: usize,
    const QUANTIZED_SIZE: usize,
    CentroidComparator,
    QuantizedComparator,
    FullComparator,
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
    fn store(&mut self, i: Box<dyn Iterator<Item = Self::T>>) -> Vec<VectorId>;
}

pub trait CentroidComparatorConstructor: Comparator {
    fn new(centroids: Vec<Self::T>) -> Self;
}

pub trait QuantizedComparatorConstructor: Comparator {
    type CentroidComparator: Comparator;

    fn new(cc: &Self::CentroidComparator) -> Self;
}

impl<
        const SIZE: usize,
        const CENTROID_SIZE: usize,
        const QUANTIZED_SIZE: usize,
        CentroidComparator: Comparator<T = [f32; CENTROID_SIZE]> + CentroidComparatorConstructor + 'static,
        QuantizedComparator: Comparator<T = [u16; QUANTIZED_SIZE]>
            + VectorStore<T = [u16; QUANTIZED_SIZE]>
            + PartialDistance
            + QuantizedComparatorConstructor<CentroidComparator = CentroidComparator>
            + 'static,
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
    pub fn new(selection_size: usize, comparator: FullComparator) -> Self {
        let vector_selection = comparator.selection(selection_size);
        // Linfa
        let data: Vec<f32> = vector_selection
            .into_iter()
            .flat_map(|v| v.into_iter())
            .collect();
        let sub_length = data.len() / CENTROID_SIZE;
        let sub_arrays = Array::from_shape_vec((sub_length, CENTROID_SIZE), data).unwrap();
        eprintln!("sub_arrays: {sub_arrays:?}");
        let observations = DatasetBase::from(sub_arrays);
        // TODO review this number
        let number_of_clusters = usize::min(sub_length, 1_000);
        let prng = StdRng::seed_from_u64(42);

        eprintln!("{} Running kmeans", Utc::now());
        let model = KMeans::params_with_rng(number_of_clusters, prng.clone())
            .tolerance(1e-2)
            .fit(&observations)
            .expect("KMeans fitted");
        eprintln!("{} kmeans finished", Utc::now());
        let centroid_array: Array2<f32> = model.centroids().clone();
        centroid_array.len();
        let centroid_flat: Vec<f32> = centroid_array
            .into_shape(number_of_clusters * CENTROID_SIZE)
            .unwrap()
            .to_vec();
        eprintln!("centroid flat len: {}", centroid_flat.len());
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

        let vector_ids = (0..centroids.len()).map(VectorId).collect();
        let centroid_comparator = CentroidComparator::new(centroids);
        let centroid_m = 24;
        let centroid_m0 = 48;
        let centroid_order = 12;
        let mut quantized_comparator = QuantizedComparator::new(&centroid_comparator);
        let mut centroid_hnsw: Hnsw<CentroidComparator> = Hnsw::generate(
            centroid_comparator,
            vector_ids,
            centroid_m,
            centroid_m0,
            centroid_order,
        );
        //centroid_hnsw.improve_index();
        centroid_hnsw.improve_neighbors(0.01, 1.0);

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

    pub fn vector_count(&self) -> usize {
        self.hnsw.vector_count()
    }

    pub fn quantizer(
        &self,
    ) -> &HnswQuantizer<SIZE, CENTROID_SIZE, QUANTIZED_SIZE, CentroidComparator> {
        &self.quantizer
    }

    pub fn centroid_comparator(&self) -> &CentroidComparator {
        self.quantizer.comparator()
    }

    pub fn quantized_comparator(&self) -> &QuantizedComparator {
        self.hnsw.comparator()
    }

    pub fn full_comparator(&self) -> &FullComparator {
        &self.comparator
    }

    pub fn search(
        &self,
        v: AbstractVector<[f32; SIZE]>,
        number_of_candidates: usize,
        probe_depth: usize,
    ) -> Vec<(VectorId, f32)> {
        let raw_v = self.comparator.lookup_abstract(v.clone());
        let quantized = self.quantizer.quantize(&raw_v);
        let result = self.hnsw.search(
            AbstractVector::Unstored(&quantized),
            number_of_candidates,
            probe_depth,
        );
        let mut reordered = Vec::with_capacity(result.len());
        for (id, _) in result {
            let dist = self
                .full_comparator()
                .compare_vec(AbstractVector::Stored(id), v.clone());
            reordered.push((id, dist))
        }
        reordered.sort_by_key(|(vid, d)| (OrderedFloat(*d), *vid));
        // TODO reorder
        reordered
    }

    pub fn improve_neighbors(&mut self, threshold: f32, recall_proportion: f32) {
        self.hnsw.improve_neighbors(threshold, recall_proportion)
    }

    pub fn zero_neighborhood_size(&self) -> usize {
        self.hnsw.zero_neighborhood_size()
    }
    pub fn threshold_nn(
        &self,
        threshold: f32,
        probe_depth: usize,
        initial_search_depth: usize,
    ) -> impl IndexedParallelIterator<Item = (VectorId, Vec<(VectorId, f32)>)> + '_ {
        self.hnsw
            .threshold_nn(threshold, probe_depth, initial_search_depth)
    }
    pub fn stochastic_recall(&self, recall_proportion: f32) -> f32 {
        self.hnsw.stochastic_recall(recall_proportion)
    }
}

impl<
        const SIZE: usize,
        const CENTROID_SIZE: usize,
        const QUANTIZED_SIZE: usize,
        ComparatorParams,
        CentroidComparator: Serializable<Params = ()> + Clone + Sync + 'static,
        QuantizedComparator: Serializable<Params = ()> + Clone + 'static,
        FullComparator: Serializable<Params = ComparatorParams> + 'static,
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
        eprintln!("serializing hnsw");
        self.hnsw.serialize(hnsw_path)?;

        let comparator_path = path_buf.join("comparator");
        eprintln!("serializing comparator");
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
        let hnsw: Hnsw<QuantizedComparator> = Hnsw::deserialize(hnsw_path, ())?;

        let comparator_path = path_buf.join("comparator");
        let comparator = FullComparator::deserialize(comparator_path, params)?;

        Ok(Self {
            quantizer,
            hnsw,
            comparator,
        })
    }
}

#[cfg(test)]
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

    use crate::bigvec::random_normed_vec;

    use super::*;

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
        data: Arc<Vec<[f32; 32]>>,
    }

    impl Comparator for CentroidComparator32 {
        type T = [f32; 32];
        type Borrowable<'a> = &'a Self::T;
        fn lookup(&self, v: crate::VectorId) -> Self::Borrowable<'_> {
            &self.data[v.0]
        }

        fn compare_raw(&self, v1: &Self::T, v2: &Self::T) -> f32 {
            cosine32(v1, v2)
        }
    }

    impl CentroidComparatorConstructor for CentroidComparator32 {
        fn new(centroids: Vec<Self::T>) -> Self {
            Self {
                data: Arc::new(centroids),
            }
        }
    }

    #[derive(Clone)]
    struct QuantizedComparator32 {
        cc: CentroidComparator32,
        data: Arc<RwLock<Vec<[u16; 48]>>>,
    }

    impl PartialDistance for QuantizedComparator32 {
        fn partial_distance(&self, _i: u16, _j: u16) -> f32 {
            todo!()
        }
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
                .flat_map(|i| self.cc.lookup(VectorId(*i as usize)).into_iter().copied())
                .collect();
            let v_reconstruct2: Vec<f32> = v2
                .iter()
                .flat_map(|i| self.cc.lookup(VectorId(*i as usize)).into_iter().copied())
                .collect();
            let mut ar1 = [0.0_f32; 1536];
            let mut ar2 = [0.0_f32; 1536];
            ar1.copy_from_slice(&v_reconstruct1);
            ar2.copy_from_slice(&v_reconstruct2);
            cosine1536(&ar1, &ar2)
        }
    }

    impl QuantizedComparatorConstructor for QuantizedComparator32 {
        type CentroidComparator = CentroidComparator32;

        fn new(cc: &Self::CentroidComparator) -> Self {
            Self {
                cc: cc.clone(),
                data: Default::default(),
            }
        }
    }

    impl VectorStore for QuantizedComparator32 {
        type T = <QuantizedComparator32 as Comparator>::T;

        fn store(&mut self, i: Box<dyn Iterator<Item = Self::T>>) -> Vec<VectorId> {
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

        fn store(&mut self, i: Box<dyn Iterator<Item = Self::T>>) -> Vec<VectorId> {
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

    #[derive(Clone)]
    struct Comparator16 {
        data: Arc<RwLock<Vec<[f32; 16]>>>,
    }

    impl Comparator for Comparator16 {
        type T = [f32; 16];
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

    impl VectorStore for Comparator16 {
        type T = <Comparator16 as Comparator>::T;

        fn store(&mut self, i: Box<dyn Iterator<Item = Self::T>>) -> Vec<VectorId> {
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

    impl VectorSelector for Comparator16 {
        type T = <Comparator16 as Comparator>::T;

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

    #[derive(Clone)]
    struct QuantizedComparator4 {
        cc: CentroidComparator4,
        data: Arc<RwLock<Vec<[u16; 4]>>>,
    }

    impl QuantizedComparatorConstructor for QuantizedComparator4 {
        type CentroidComparator = CentroidComparator4;

        fn new(cc: &Self::CentroidComparator) -> Self {
            Self {
                cc: cc.clone(),
                data: Default::default(),
            }
        }
    }

    impl PartialDistance for QuantizedComparator4 {
        fn partial_distance(&self, _i: u16, _j: u16) -> f32 {
            todo!()
        }
    }

    impl Comparator for QuantizedComparator4 {
        type T = [u16; 4];
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
                .flat_map(|i| self.cc.lookup(VectorId(*i as usize)).into_iter().copied())
                .collect();
            let v_reconstruct2: Vec<f32> = v2
                .iter()
                .flat_map(|i| self.cc.lookup(VectorId(*i as usize)).into_iter().copied())
                .collect();
            v_reconstruct1
                .iter()
                .zip(v_reconstruct2.iter())
                .map(|(f1, f2)| (f1 - f2).powi(2))
                .sum::<f32>()
                .sqrt()
        }
    }

    impl VectorStore for QuantizedComparator4 {
        type T = <QuantizedComparator4 as Comparator>::T;

        fn store(&mut self, i: Box<dyn Iterator<Item = Self::T>>) -> Vec<VectorId> {
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
    struct CentroidComparator4 {
        data: Arc<Vec<[f32; 4]>>,
    }

    impl CentroidComparatorConstructor for CentroidComparator4 {
        fn new(centroids: Vec<Self::T>) -> Self {
            Self {
                data: Arc::new(centroids),
            }
        }
    }

    impl Comparator for CentroidComparator4 {
        type T = [f32; 4];
        type Borrowable<'a> = &'a Self::T;
        fn lookup(&self, v: crate::VectorId) -> Self::Borrowable<'_> {
            &self.data[v.0]
        }

        fn compare_raw(&self, v1: &Self::T, v2: &Self::T) -> f32 {
            v1.iter()
                .zip(v2.iter())
                .map(|(f1, f2)| (f1 - f2).powi(2))
                .sum::<f32>()
                .sqrt()
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
        let count = 10;
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
        let fc = AIComparator {
            data: Arc::new(RwLock::new(vecs.clone())),
        };
        let hnsw: QuantizedHnsw<1536, 32, 48, CentroidComparator32, QuantizedComparator32, _> =
            QuantizedHnsw::new(100, fc);
        let v = AbstractVector::Unstored(&vecs[0]);
        let res = hnsw.search(v, 10, 2);
        eprintln!("res: {res:?}");
        panic!();
    }

    #[test]
    fn test_small_pq() {
        let count = 10000;
        let vecs: Vec<[f32; 16]> = (0..count)
            .into_par_iter()
            .map(move |i| {
                let mut prng = StdRng::seed_from_u64(42_u64 + i as u64);
                let mut arr = [0.0_f32; 16];
                let v = random_normed_vec(&mut prng, 16);
                arr.copy_from_slice(&v);
                arr
            })
            .collect();
        let fc = Comparator16 {
            data: Arc::new(RwLock::new(vecs.clone())),
        };
        let mut hnsw: QuantizedHnsw<16, 4, 4, CentroidComparator4, QuantizedComparator4, _> =
            QuantizedHnsw::new(100, fc);
        hnsw.improve_neighbors(0.01, 1.0);

        // Test last vector individually
        let raw_vec = vecs.last().unwrap();
        eprintln!("raw_vec: {raw_vec:?}");
        let quant_vec = hnsw.quantizer.quantize(raw_vec);
        eprintln!("quant_vec: {quant_vec:?}");
        let recon_vec = hnsw.quantizer.reconstruct(&quant_vec);
        eprintln!("recon_vec: {recon_vec:?}");
        let lvid = VectorId(vecs.len() - 1);
        let internal_quantized = *hnsw.quantized_comparator().lookup(lvid);
        eprintln!("internal quant: {internal_quantized:?}");
        let av = AbstractVector::Unstored(raw_vec);
        let res = hnsw.search(av, 10, 2);
        eprintln!("Match results: {res:?}");

        let mut matches = 0;
        let mut match_sum = 0.0;
        for (i, v) in vecs.iter().enumerate() {
            let av = AbstractVector::Unstored(v);
            let res = hnsw.search(av, 30, 2);
            if let Some((matchvid, match_distance)) = res.first() {
                if i == matchvid.0 {
                    matches += 1;
                    match_sum += match_distance;
                } else {
                    eprintln!("Match result for {i:?} {res:?}");
                }
            }
        }
        let recall = matches as f32 / vecs.len() as f32;
        eprintln!("recall: {recall}");
        let match_avg = match_sum / vecs.len() as f32;
        eprintln!("average match distance: {match_avg}");
        panic!();
    }
}
