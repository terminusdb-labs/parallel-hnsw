use std::{path::Path, sync::Arc};

use crate::{AbstractVector, Comparator, Hnsw, SerializationError, VectorId};

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Uniform;
use rayon::prelude::*;

pub fn make_random_hnsw(count: usize, dimension: usize) -> Hnsw<BigComparator, BigVec> {
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
    let hnsw: Hnsw<BigComparator, BigVec> = Hnsw::generate(c, vs, m, m0);
    hnsw
}

pub type BigVec = Vec<f32>;
#[derive(Clone, Debug, PartialEq)]
pub struct BigComparator {
    pub data: Arc<Vec<BigVec>>,
}

impl Comparator<BigVec> for BigComparator {
    type Params = ();
    fn compare_vec(&self, v1: AbstractVector<BigVec>, v2: AbstractVector<BigVec>) -> f32 {
        let v1 = match v1 {
            AbstractVector::Stored(i) => &self.data[i.0],
            AbstractVector::Unstored(v) => v,
        };
        let v2 = match v2 {
            AbstractVector::Stored(i) => &self.data[i.0],
            AbstractVector::Unstored(v) => v,
        };
        let mut result = 0.0;
        for (&f1, &f2) in v1.iter().zip(v2.iter()) {
            result += f1 * f2
        }
        1.0 - result
    }

    fn serialize<P: AsRef<Path>>(&self, _path: P) -> Result<(), SerializationError> {
        Ok(())
    }

    fn deserialize<P: AsRef<Path>>(_path: P, _: ()) -> Result<BigComparator, SerializationError> {
        Ok(BigComparator {
            data: Arc::new(Vec::new()),
        })
    }
}

fn random_normed_vec(prng: &mut StdRng, size: usize) -> Vec<f32> {
    let range = Uniform::from(0.0..1.0);
    let vec: Vec<f32> = prng.sample_iter(&range).take(size).collect();
    let norm = vec.iter().map(|f| f * f).sum::<f32>().sqrt();
    let res = vec.iter().map(|f| f / norm).collect();
    res
}
