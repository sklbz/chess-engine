use chess::board::board::Board;
use chess::board::nnue_input_vector::VectorOutput;
use chess::legal_moves::misc::Color;
use chess::utils::move_to_string;
use multilayer_perceptron::mlp::io::Save;
use multilayer_perceptron::mlp::multilayer_perceptron::*;
use multilayer_perceptron::mlp::utils::Database;
use rand::distr::Distribution;

use super::distribution::{Display, ProbabilityDistribution};
use super::relu::ReLU;
use super::softmax::Softmax;

use crate::utils::move_to_number::move_hash;
use crate::utils::move_to_output::move_vec;
use crate::utils::number_to_move::move_from;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

pub struct ChessEngine {
    mlp: MultiLayerPerceptron,
}

impl ChessEngine {
    pub fn new() -> ChessEngine {
        // Inspiration from Stockfish NNUE architecture
        let architecture = vec![768, 1024, 1536, 1792];
        // let simple_architecture = vec![768, 1792];

        let engine = MultiLayerPerceptron::new(architecture);

        ChessEngine { mlp: engine }
    }

    pub fn predict(&self, board: &Board, color: &Color) -> String {
        let input: Vec<f64> = board.to_vector();

        // println!("Input: {:?}", input);

        let raw_output: Vec<f64> = self.mlp.calc(input);

        // println!("Raw output: {:?}", raw_output);

        let lower_bound = 0.1f64.ln();
        let upper_bound = f64::MAX.ln();

        let bounded_output: Vec<f64> = raw_output
            .iter()
            .map(|x| x.clamp(lower_bound, upper_bound))
            .collect();

        let _rectified_output: Vec<f64> = bounded_output.relu();

        let legal_moves: Vec<String> = board
            .get_legal_moves(color)
            .iter()
            .map(move_to_string)
            .collect();
        let moves_indices = legal_moves
            .iter()
            .map(|mv| move_hash(mv))
            .collect::<Vec<usize>>();

        let temperature = 5.0;

        let trimmed_output = bounded_output
            .iter()
            .enumerate()
            .filter(|(i, _): &(usize, &f64)| moves_indices.contains(i))
            .map(|(_, x)| *x)
            .collect::<Vec<f64>>()
            .softmax(temperature);

        let distribution = ProbabilityDistribution::new(moves_indices, trimmed_output, legal_moves);
        //DEBUG-----------------------------------------------------------------------------------
        distribution.display();
        //----------------------------------------------------------------------------------------
        let move_index = distribution.sample(&mut rand::rng());

        move_from(move_index).to_string()
    }

    pub fn train_from_file(&mut self, file_path: &str) {
        let mut data: Database = Vec::new();

        let path = Path::new(file_path);
        let file = File::open(path).expect("no such file");
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let mut board = Board::init();

            let line = line.unwrap();
            line.split_whitespace().for_each(|x| {
                data.push((board.to_vector(), move_vec(x.to_string()), 1.0));
                board.make_move_str(x);
            })
        }

        println!("Training with {} examples", data.len());

        let start = std::time::Instant::now();

        self.mlp.backpropagation(data, 1, 0.1);

        let training_time = start.elapsed().as_secs();
        let minutes = training_time / 60;
        let seconds = training_time % 60;
        println!("Training took {} minutes and {} seconds", minutes, seconds);
    }

    pub fn save_model(&self, file_path: &str) {
        println!("Saving model to {}", file_path);
        self.mlp.save(file_path.to_string());
    }

    pub fn load_model(&mut self, file_path: &str) {
        println!("Loading model from {}", file_path);
        self.mlp = MultiLayerPerceptron::load(file_path.to_string());
    }

    pub fn load_params(&mut self, file_path: &str) {
        println!("Loading parameters from {}", file_path);
        self.mlp.load_params(file_path);
    }

    pub fn params(&self) -> String {
        self.mlp.params()
    }
}
