use std::path::Path;
use std::sync::Arc;

use linfa::traits::Fit;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use ndarray::Array;
use parallel_hnsw::bigvec::{BigComparator, BigVec};
use parallel_hnsw::serialize::SerializationError;
use parallel_hnsw::{AbstractVector, Comparator, Hnsw, VectorId};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Uniform;
use rayon::prelude::*;

// assumes normalized vectors
fn cosine(v1: &Vec<f32>, v2: &Vec<f32>) -> f32 {
    (1.0 - v1
        .iter()
        .zip(v2.iter())
        .map(|(f1, f2)| f1 * f2)
        .sum::<f32>())
        / 2.0
}

pub type QuantizedVec = Vec<usize>;

#[derive(Clone)]
pub struct CentroidComparator {
    centroid_size: usize,
    centroids: Arc<Vec<BigVec>>,
    data: Arc<Vec<QuantizedVec>>,
    hnsw: Arc<Hnsw<BigComparator>>,
}

impl CentroidComparator {
    fn reconstruct(&self, vin: &QuantizedVec) -> Vec<f32> {
        let size = self.centroid_size * vin.len();
        let mut v = Vec::with_capacity(size);
        for i in vin {
            v.extend(self.centroids[*i].iter())
        }
        v
    }

    fn quantize(&self, vin: &Vec<f32>) -> QuantizedVec {
        let len = vin.len();
        let parts = len / self.centroid_size;
        assert_eq!(len % self.centroid_size, 0);
        let mut vec: Vec<usize> = Vec::with_capacity(parts);
        for v in vin.chunks(parts) {
            let distances = self
                .hnsw
                .search(AbstractVector::Unstored(&v.to_vec()), 100, 2);
            vec.push(distances[0].0 .0)
        }
        vec
    }
}
impl Comparator for CentroidComparator {
    type Params = ();
    type T = QuantizedVec;
    fn lookup(&self, v: VectorId) -> &QuantizedVec {
        &self.data[v.0]
    }

    fn compare_raw(&self, v1: &Self::T, v2: &Self::T) -> f32 {
        cosine(&self.reconstruct(v1), &self.reconstruct(v2))
    }
}

fn random_normed_vec(prng: &mut StdRng, size: usize) -> Vec<f32> {
    let range = Uniform::from(0.0..1.0);
    let vec: Vec<f32> = prng.sample_iter(&range).take(size).collect();
    let norm = vec.iter().map(|f| f * f).sum::<f32>().sqrt();
    let res = vec.iter().map(|f| f / norm).collect();
    res
}

fn do_test_recall(hnsw: &Hnsw<CentroidComparator>) -> f32 {
    let data = &hnsw.layers[0].comparator.data;
    let total = data.len();
    let total_relevant: usize = data
        .par_iter()
        .enumerate()
        .map(|(i, datum)| {
            /* eprintln!("XXXXXXXXXXXXXXXXXXXXXX");
            eprintln!("Searching for {i}");
             */
            let v = AbstractVector::Unstored(datum);
            let results = hnsw.search(v, 300, 2);
            if VectorId(i) == results[0].0 {
                1
            } else {
                0
            }
        })
        .sum();
    eprintln!("total relevant: {total_relevant}");
    eprintln!("from total: {total}");
    let recall = total_relevant as f32 / total as f32;
    eprintln!("with recall: {recall}");

    recall
}

pub fn main() {
    let count = 1000;
    let clusters = 1000;
    let dimension = 1536; // 32 * 48
    let sub_dimension = 32;
    let sub_length = dimension * count / sub_dimension;
    let mut prng = StdRng::seed_from_u64(42);
    let data: Vec<f32> = random_normed_vec(&mut prng, dimension * count);
    let sub_arrays = Array::from_shape_vec((sub_dimension, sub_length), data).unwrap();
    let observations = DatasetBase::from(sub_arrays.clone());
    eprintln!("Starting clustering");
    let model = KMeans::params_with_rng(clusters, prng.clone())
        .tolerance(1e-2)
        .fit(&observations)
        .expect("KMeans fitted");
    eprintln!("centroids discovered");
    let centroid_flat: Vec<f32> = model.centroids().clone().into_raw_vec();
    let centroids: Vec<Vec<f32>> = centroid_flat
        .chunks(sub_dimension)
        .map(|v| v.to_vec())
        .collect();
    let vectors: Vec<_> = (0..centroids.len()).map(VectorId).collect();
    let c = BigComparator {
        data: Arc::new(centroids.clone()),
    };
    let m = 24;
    let m0 = 48;
    let order = 24;
    let hnsw: Hnsw<BigComparator> = Hnsw::generate(c, vectors, m, m0, order);
    let vec_number = 1_000_000;
    let vecs: Vec<Vec<f32>> = (0..vec_number)
        .into_par_iter()
        .map(move |i| {
            let mut prng = StdRng::seed_from_u64(42_u64 + i as u64);
            random_normed_vec(&mut prng, dimension)
        })
        .collect();
    let vec_ids: Vec<_> = (0..vecs.len()).map(VectorId).collect();
    let mut cc = CentroidComparator {
        data: Arc::new(Vec::new()),
        hnsw: Arc::new(hnsw),
        centroids: Arc::new(centroids),
        centroid_size: sub_dimension,
    };
    eprintln!("Quantizing vectors");
    let qvecs: Vec<QuantizedVec> = vecs.par_iter().map(|v| cc.quantize(v)).collect();
    let avg_reconstruction_cost = qvecs
        .iter()
        .zip(vecs.iter())
        .par_bridge()
        .map(|(qv, v)| cosine(v, &cc.reconstruct(qv)))
        .sum::<f32>()
        / qvecs.len() as f32;
    eprintln!("Average reconstruction cost: {avg_reconstruction_cost}");
    cc.data = Arc::new(qvecs);
    let mut qhnsw: Hnsw<CentroidComparator> = Hnsw::generate(cc, vec_ids, m, m0, order);
    let initial_recall = do_test_recall(&qhnsw);
    let mut last_recall = initial_recall;
    let mut improvement = f32::MAX;
    eprintln!("Improving index");
    while improvement > 0.001 {
        qhnsw.improve_index();
        let new_recall = do_test_recall(&qhnsw);
        improvement = new_recall - last_recall;
        last_recall = new_recall;
    }
}
