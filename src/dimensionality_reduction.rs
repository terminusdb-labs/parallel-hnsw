use rand::{rngs::StdRng, Rng, SeedableRng};

/*

random "matrix" for dimensionality reduction

*/

#[derive(Debug)]
pub struct ProjectionMatrix {
    pub k: usize,
    pub d: usize,
    pub words: Vec<usize>,
}

/*

        (1 * 1) *2
          2
0,0 0,1 | 1,0  1,1
01  10  |  00  11
 */

impl ProjectionMatrix {
    pub fn get(&self, i: usize, j: usize) -> f32 {
        let idx = (i + (j * i)) * 2;
        let word_offset: usize = idx / usize::BITS as usize;
        let word = self.words[word_offset];
        let word_idx: usize = idx % usize::BITS as usize;
        decode(word >> word_idx) as f32 * f32::sqrt(3.0)
    }

    pub fn project(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), self.d);
        let mut w: Vec<f32> = Vec::with_capacity(self.k);
        for j in 0..self.k {
            let mut w_j = 0_f32;
            for (i, v_i) in v.iter().enumerate() {
                let f = self.get(i, j);
                w_j += f * v_i;
            }
            w.push(w_j)
        }
        w
    }

    pub fn new(d: usize, k: usize) -> ProjectionMatrix {
        let dimension = k * d;
        let mut prng = StdRng::seed_from_u64(dimension as u64);
        let word_count = dimension * 2 / usize::BITS as usize + 1;
        let mut words: Vec<usize> = Vec::with_capacity(word_count);
        let mut word = 0;
        for i in 0..dimension {
            let r = prng.gen_range(0..6);
            let value = if r == 0 {
                1
            } else if r == 5 {
                -1
            } else {
                0
            };
            let bits = encode(value);
            word += bits;
            if (i + 1_usize) % (usize::BITS as usize / 2_usize) == 0 {
                words.push(word);
                word = 0;
            } else {
                word <<= 2;
            }
        }
        if words.len() < word_count {
            words.push(word);
        }
        Self { words, k, d }
    }
}

// encodes -1, 0, 1 as a bit sequence (of 2 bits)
fn encode(v: i64) -> usize {
    v.saturating_add_unsigned(1) as usize
}

// decodes 0, 1, 2 as a 0, 1, -1
fn decode(v: usize) -> i64 {
    // forget higher order bits
    (v & 3) as i64 - 1
}

#[cfg(test)]
mod tests {
    use crate::dimensionality_reduction::ProjectionMatrix;

    #[test]
    fn small_d_k() {
        let d = 9;
        let k = 3;
        let m = ProjectionMatrix::new(d, k);
        let v: &[f32; 9] = &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let r = m.project(v);
        assert_eq!(r, &[-2.0, 0.0, 1.0]);
    }

    #[test]
    fn two_words_d_k() {
        let d = 12;
        let k = 4;
        let m = ProjectionMatrix::new(d, k);
        let v: &[f32; 12] = &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let r = m.project(v);
        assert_eq!(r, [0.0, 1.0, 3.0, -3.0]);
    }
}
