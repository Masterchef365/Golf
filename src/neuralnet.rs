use rand::distributions::Uniform;
use rand::{thread_rng, Rng};
//use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone)]//, Serialize, Deserialize)]
struct Layer {
    weights: Box<[f32]>,
    biases: Box<[f32]>,
    input_size: usize,
    output_size: usize,
    //#[serde(skip)]
    out_buf: Box<[f32]>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let size = input_size * output_size;
        let unif = Uniform::new(-1.0, 1.0);
        let rng = thread_rng();
        Self {
            weights: rng.sample_iter(&unif).take(size).collect(),
            biases: rng.sample_iter(&unif).take(size).collect(),
            input_size,
            output_size,
            out_buf: vec![0.0; output_size].into(),
        }
    }

    pub fn fuzz(&mut self, learning_rate: f32) {
        let unif = Uniform::new(-learning_rate, learning_rate);
        let rng = thread_rng();
        for (v, d) in self
            .weights
            .iter_mut()
            .chain(self.biases.iter_mut())
            .zip(rng.sample_iter(&unif))
        {
            *v += d;
        }
    }

    pub fn infer(&mut self, input: &[f32]) -> &[f32] {
        if self.out_buf.len() != self.output_size {
            self.out_buf = vec![0.0; self.output_size].into();
        }

        for ((weight_row, bias_row), out) in self
            .weights
            .chunks_exact(self.input_size)
            .zip(self.biases.chunks_exact(self.input_size))
            .zip(self.out_buf.iter_mut())
        {
            *out = 0.0;
            for ((weight, bias), input) in weight_row.iter().zip(bias_row.iter()).zip(input.iter())
            {
                *out += weight * input + bias
            }
        }
        &self.out_buf
    }
}

#[derive(Debug, Clone)]//, Serialize, Deserialize)]
pub struct NeuralNet {
    hidden_0: Layer,
    hidden_1: Layer,
    hidden_2: Layer,
}

impl NeuralNet {
    pub fn new() -> Self {
        Self {
            hidden_0: Layer::new(97, 10),
            hidden_1: Layer::new(10, 9),
            hidden_2: Layer::new(9, 9),
        }
    }

    pub fn infer(&mut self, input_layer: &[f32]) -> &[f32] {
        let l0 = self.hidden_0.infer(input_layer);
        let l1 = self.hidden_1.infer(l0);
        self.hidden_2.infer(l1)
    }

    pub fn fuzz(&mut self, learning_rate: f32) {
        self.hidden_0.fuzz(learning_rate);
        self.hidden_1.fuzz(learning_rate);
        self.hidden_2.fuzz(learning_rate);
    }

    /*
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;
        bincode::serialize_into(&mut file, self)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        Ok(bincode::deserialize_from(&mut file)?)
    }
    */
}

