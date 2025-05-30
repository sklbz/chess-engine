use chess::board::board::Board;
use chess::legal_moves::generate_possible_moves::generate_move_vec;
use multilayer_perceptron::mlp::multilayer_perceptron::MultilayerPerceptron;

fn main() {
    let mut board = Board::init();
    // Inspiration from Stockfish NNUE architecture
    let architecture = vec![256, 512, 1024, 1792];

    let mut engine = MultilayerPerceptron::new(architecture);
}
