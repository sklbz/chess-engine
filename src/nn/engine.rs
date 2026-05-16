use chess::board::board::Board;
use chess::board::nnue_input_vector::VectorOutput;
use chess::legal_moves::misc::Color;
use chess::utils::flip_move;
use chess::utils::move_to_string;
use multilayer_perceptron::mlp::multilayer_perceptron::MultiLayerPerceptron;
use multilayer_perceptron::mlp::multilayer_perceptron::NeuralNetwork;
use multilayer_perceptron::mlp::utils::Database;
use rand::distr::Distribution;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;

use super::distribution::{Display, ProbabilityDistribution};
use super::relu::ReLU;
use super::softmax::Softmax;

use crate::database::load_db;
use crate::database::save_db;
use crate::utils::move_to_number::move_hash;
use crate::utils::move_to_output::move_vec;
use crate::utils::number_to_move::move_from;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::io::stdout;
use std::path::Path;

pub struct ChessEngine {
    mlp: MultiLayerPerceptron,
    temp: f64,
}

impl ChessEngine {
    pub fn new() -> ChessEngine {
        // Inspiration from Stockfish NNUE architecture
        // let architecture = vec![768, 1024, 1536, 1792];
        // let simple_architecture = vec![768, 1792];
        let architecture = vec![768, 1024, 1536, 1792];

        let engine = MultiLayerPerceptron::new(architecture);

        ChessEngine {
            mlp: engine,
            temp: 1.0,
        }
    }

    pub fn get_temp(&self) -> f64 {
        self.temp
    }

    pub fn set_temp(&mut self, temp: f64) -> f64 {
        self.temp = temp.clamp(1., 10.);
        self.temp
    }

    pub fn temp_up(&mut self) -> f64 {
        self.set_temp(self.temp + 1.)
    }

    pub fn temp_down(&mut self) -> f64 {
        self.set_temp(self.temp - 1.)
    }

    pub fn predict_symm(&self, board: &Board, color: &Color) -> String {
        // Toujours du point de vue du joueur actif
        let input: Vec<f64> = board.to_vector_for(color);

        let raw_output: Vec<f64> = self.mlp.calc(&input);
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
            .map(|mv| match color {
                Color::Black => flip_move(move_to_string(mv)),
                _ => move_to_string(mv),
            })
            .collect();
        let moves_indices = legal_moves
            .iter()
            .map(|mv| move_hash(mv))
            .collect::<Vec<usize>>();

        let trimmed_output = bounded_output
            .iter()
            .enumerate()
            .filter(|(i, _): &(usize, &f64)| moves_indices.contains(i))
            .map(|(_, x)| *x)
            .collect::<Vec<f64>>()
            .softmax(self.temp);

        let distribution = ProbabilityDistribution::new(moves_indices, trimmed_output, legal_moves);
        let move_index = distribution.sample(&mut rand::rng());

        let chosen = move_from(move_index).to_string();
        match color {
            Color::Black => flip_move(chosen),
            _ => chosen,
        }
    }

    pub fn predict(&self, board: &Board, color: &Color) -> String {
        let input: Vec<f64> = board.to_vector();

        let raw_output: Vec<f64> = self.mlp.calc(&input);

        for (i, val) in raw_output.iter().enumerate() {
            if val.is_nan() || val.is_infinite() {
                println!("Problem at index {i}: {val}");
            }
        }

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

        let trimmed_output = bounded_output
            .iter()
            .enumerate()
            .filter(|(i, _): &(usize, &f64)| moves_indices.contains(i))
            .map(|(_, x)| *x)
            .collect::<Vec<f64>>()
            .softmax(self.temp);

        let distribution = ProbabilityDistribution::new(moves_indices, trimmed_output, legal_moves);
        // DEBUG----------------------------------------------------------------------------------
        // distribution.display();
        //----------------------------------------------------------------------------------------
        let move_index = distribution.sample(&mut rand::rng());

        move_from(move_index).to_string()
    }

    pub fn train_from_samples(&mut self, data: &Database) {
        let cycle_amount = 1;
        for i in 0..cycle_amount {
            let cycle_start = std::time::Instant::now();

            self.mlp.backpropagation(data, 1, 0.005);

            let cycle_time = cycle_start.elapsed().as_secs();
            let cycle_minutes = cycle_time / 60;
            let cycle_seconds = cycle_time % 60;

            println!("Cycle {i} done, took {cycle_minutes} minutes and {cycle_seconds} seconds");
        }
    }

    pub fn auto_train(&mut self, file_path: &str) {
        let db_path = format!("{file_path}.db");

        let data_result = match Path::new(&db_path).is_file() {
            true => load_db(&db_path),
            false => save_db(file_path),
        };

        if let Ok(data) = data_result {
            println!("Training with {} examples", data.len());
            loop {
                let start = std::time::Instant::now();
                let learning_rate = 0.000075;
                self.mlp.backpropagation(&data, 200, learning_rate);

                let training_time = start.elapsed().as_secs();
                let minutes = training_time / 60;
                let seconds = training_time % 60;

                println!();
                println!("Training cycle took {minutes} minutes and {seconds} seconds");
                self.save_model();
            }
        }
    }

    pub fn train_from_file_symm(&mut self, file_path: &str) {
        let db_path = format!("{file_path}.db");

        let data_result = match Path::new(&db_path).is_file() {
            true => load_db(&db_path),
            false => save_db(file_path),
        };

        if let Ok(data) = data_result {
            println!("Training with {} examples", data.len());

            let start = std::time::Instant::now();
            let learning_rate = 0.0001;
            self.mlp.backpropagation(&data, 100, learning_rate);

            let training_time = start.elapsed().as_secs();
            let minutes = training_time / 60;
            let seconds = training_time % 60;

            println!("\rTraining took {minutes} minutes and {seconds} seconds");
        }
    }

    pub fn train_from_file(&mut self, file_path: &str) {
        let mut data: Database = Vec::new();

        let path = Path::new(file_path);
        let file = File::open(path).expect("no such file");
        let reader = BufReader::new(file);

        for line in reader.lines() {
            stdout().flush().unwrap();

            let mut board = Board::init();

            let line = line.unwrap();
            line.split_whitespace().for_each(|x| {
                data.push((
                    board.to_vector().into(),
                    move_vec(x.to_string()).into(),
                    1.0,
                ));
                board.make_move_str(x);
            })
        }

        let n = data.len() as f64;
        data.iter_mut().for_each(|(_, _, c)| *c = 1.0 / n);

        // print!("Desired cycle amount : ");

        /*
        let cycle_amount = user_input()
        .split_whitespace()
        .next()
        .unwrap()
        .parse::<usize>()
        .expect("unable to parse number of cycles");*/

        println!("Training with {} examples", data.len());

        let start = std::time::Instant::now();
        let learning_rate = 0.00005;
        self.mlp.backpropagation(&data, 100, learning_rate);
        /*
        for i in 0..=cycle_amount {
            let cycle_start = std::time::Instant::now();

            let use_data: Vec<(Vec<f64>, Vec<f64>, f64)> = data.clone();
            self.mlp.backpropagation(use_data, 50, 0.001);

            let cycle_time = cycle_start.elapsed().as_secs();
            let cycle_minutes = cycle_time / 60;
            let cycle_seconds = cycle_time % 60;

            println!("Cycle {i} done, took {cycle_minutes} minutes and {cycle_seconds} seconds");
        }
        */

        let training_time = start.elapsed().as_secs();
        let minutes = training_time / 60;
        let seconds = training_time % 60;

        println!();
        println!("Training took {minutes} minutes and {seconds} seconds");
    }

    pub fn save_model(&self) {
        let _ = self.mlp.save().unwrap();
    }

    pub fn load_model(&mut self, file_path: &str) {
        println!("Loading model from {file_path}");
        self.mlp = MultiLayerPerceptron::load(file_path);
    }
}
