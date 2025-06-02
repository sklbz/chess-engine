use chess::board::board::Board;
use chess::board::nnue_input_vector::VectorOutput;
use chess::legal_moves::misc::Color;
use chess::utils::move_to_string;
use multilayer_perceptron::mlp::multilayer_perceptron::*;
use rand::distr::Distribution;

use super::distribution::{Display, ProbabilityDistribution};
use super::relu::ReLU;
use super::softmax::Softmax;

use crate::move_to_number::move_hash;
use crate::number_to_move::move_from;

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

        let raw_output: Vec<f64> = self.mlp.calc(input).relu();

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

        let distribution = ProbabilityDistribution::new(moves_indices, trimmed_output);
        let move_index = distribution.sample(&mut rand::rng());

        move_from(move_index).to_string()
    }
}
