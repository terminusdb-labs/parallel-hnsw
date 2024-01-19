use std::{path::Path, sync::Arc};

use crate::{
    types::{AbstractVector, VectorId},
    Comparator, Hnsw, SerializationError,
};

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Uniform;
use rayon::prelude::*;

pub fn make_random_hnsw(count: usize, dimension: usize) -> Hnsw<BigComparator> {
    let order = 24;
    make_random_hnsw_with_order(count, dimension, order)
}

pub fn make_random_hnsw_with_order(
    count: usize,
    dimension: usize,
    order: usize,
) -> Hnsw<BigComparator> {
    let data: Vec<Vec<f32>> = (0..count)
        .into_par_iter()
        .map(move |i| {
            let mut prng = StdRng::seed_from_u64(42_u64 + i as u64);
            random_normed_vec(&mut prng, dimension)
        })
        .collect();
    let c = BigComparator {
        data: Arc::new(data),
    };
    let vs: Vec<_> = (0..count).map(VectorId).collect();
    let m = 24;
    let m0 = 48;
    let hnsw: Hnsw<BigComparator> = Hnsw::generate(c, vs, m, m0, order);
    hnsw
}

pub type BigVec = Vec<f32>;
#[derive(Clone, Debug, PartialEq)]
pub struct BigComparator {
    pub data: Arc<Vec<BigVec>>,
}

impl Comparator for BigComparator {
    type Params = ();
    type T = BigVec;
    type Borrowable<'a> = &'a BigVec;
    fn compare_raw(&self, v1: &BigVec, v2: &BigVec) -> f32 {
        let mut result = 0.0;
        for (&f1, &f2) in v1.iter().zip(v2.iter()) {
            result += f1 * f2
        }
        (1.0_f32 - result) / 2.0_f32
    }
    fn lookup(&self, v: VectorId) -> &BigVec {
        &self.data[v.0]
    }
}

fn random_normed_vec(prng: &mut StdRng, size: usize) -> Vec<f32> {
    let range = Uniform::from(0.0..1.0);
    let vec: Vec<f32> = prng.sample_iter(&range).take(size).collect();
    let norm = vec.iter().map(|f| f * f).sum::<f32>().sqrt();
    let res = vec.iter().map(|f| f / norm).collect();
    res
}
