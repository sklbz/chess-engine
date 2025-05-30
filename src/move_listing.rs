use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::move_to_number_conversion::generate_all_possible_move;

#[allow(dead_code)]
pub fn listing() {
    let args: Vec<String> = env::args().collect();
    let mut output = File::create(Path::new(&args[1])).unwrap();

    let moves = generate_all_possible_move();

    moves.iter().for_each(|mv| {
        writeln!(output, "{}", mv).unwrap();
    });
}
