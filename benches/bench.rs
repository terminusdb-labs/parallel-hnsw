#![feature(test)]
extern crate test;

use std::sync::Arc;

use parallel_hnsw::{AbstractVector, Comparator, Hnsw, VectorId};
use rand::{thread_rng, Rng};
use test::Bencher;
type SillyVec = [f32; 100];
#[derive(Clone)]
struct SillyComparator {
    data: Arc<Vec<SillyVec>>,
}

impl Comparator<SillyVec> for SillyComparator {
    type Params = ();
    fn compare_vec(&self, v1: AbstractVector<SillyVec>, v2: AbstractVector<SillyVec>) -> f32 {
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
    fn deserialize<P: AsRef<std::path::Path>>(
        _path: P,
        _params: Self::Params,
    ) -> Result<Self, parallel_hnsw::SerializationError> {
        todo!();
    }
    fn serialize<P: AsRef<std::path::Path>>(
        &self,
        _path: P,
    ) -> Result<(), parallel_hnsw::SerializationError> {
        todo!();
    }
}

fn generate_random_vector() -> SillyVec {
    let mut rng = thread_rng();

    let mut result: SillyVec = [0.0; 100];
    (0..result.len()).for_each(|i| {
        result[i] = rng.gen();
    });

    result
}

fn create_test_data(length: usize) -> SillyComparator {
    let mut vec = Vec::new();
    for _ in 0..length {
        vec.push(generate_random_vector());
    }
    SillyComparator {
        data: Arc::new(vec),
    }
}

#[bench]
fn bla(b: &mut Bencher) {
    const LENGTH: usize = 10000;
    let comparator = create_test_data(LENGTH);
    let vs: Vec<VectorId> = (0..LENGTH).map(VectorId).collect();

    b.iter(|| {
        let _result: Hnsw<_, _> = Hnsw::generate(comparator.clone(), vs.clone(), 24, 48);
    });
}
