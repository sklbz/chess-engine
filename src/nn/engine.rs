use chess::board::board::Board;
use chess::board::nnue_input_vector::VectorOutput;
use chess::legal_moves::misc::Color;
use chess::utils::move_to_string;
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
        // let architecture = vec![768, 1024, 1536, 1792];
        let simple_architecture = vec![768, 1792];

        let engine = MultiLayerPerceptron::new(simple_architecture);

        ChessEngine { mlp: engine }
    }

    pub fn predict(&self, board: &Board, color: &Color) -> String {
        let input: Vec<f64> = board.to_vector();

        let raw_output: Vec<f64> = self.mlp.calc(input);

        let _rectified_output: Vec<f64> = raw_output.relu();

        let legal_moves: Vec<String> = board
            .get_legal_moves(color)
            .iter()
            .map(move_to_string)
            .collect();
        let moves_indices = legal_moves
            .iter()
            .map(|mv| move_hash(mv))
            .collect::<Vec<usize>>();

        let trimmed_output = raw_output
            .iter()
            .enumerate()
            .filter(|(i, _)| moves_indices.contains(i))
            .map(|(_, x)| *x)
            .collect::<Vec<f64>>()
            .softmax(1.0);

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

        self.mlp.backpropagation(data, 10, 0.1);
    }
}
