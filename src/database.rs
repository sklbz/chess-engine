use std::{
    error::Error,
    fs::{File, read},
    io::{BufRead, BufReader, Write, stdout},
    path::Path,
};

use bincode::config::standard;
use chess::{
    board::{board::Board, nnue_input_vector::VectorOutput},
    legal_moves::misc::Color,
    utils::flip_move,
};
use multilayer_perceptron::mlp::utils::Database;

use Color::*;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

use crate::utils::move_to_output::move_vec;

pub fn load_db(file_path: &str) -> Result<Database, Box<dyn Error>> {
    let bytes = read(file_path)?;

    let (database, _): (Database, usize) = bincode::decode_from_slice(&bytes, standard())?;

    Ok(database)
}

pub fn save_db(file_path: &str) -> Result<Database, Box<dyn Error>> {
    let mut data: Database = Vec::new();

    let path = Path::new(file_path);
    let file = File::open(path).expect("no such file");
    let reader = BufReader::new(file);

    for line in reader.lines() {
        stdout().flush().unwrap();

        let mut board = Board::init();

        let line = line.unwrap();
        let color = White;
        line.split_whitespace().for_each(|x| {
            let input = board.to_vector_for(&color);
            let normalized_move = match color {
                Black => flip_move(x.to_string()),
                _ => x.to_string(),
            };

            data.push((input.into(), move_vec(normalized_move).into(), 1.0));
            board.make_move_str(x);
            let color = !color;
        })
    }

    let n = data.len() as f64;
    data.par_iter_mut().for_each(|(_, _, c)| *c = 1.0 / n);

    let encoded = bincode::encode_to_vec(&data, bincode::config::standard())?;
    let db_path = format!("{file_path}.db");

    let mut db_file = File::create(db_path)?;
    db_file.write_all(&encoded)?;

    Ok(data)
}
