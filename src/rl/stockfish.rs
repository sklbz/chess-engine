use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

#[derive(Debug)]
enum Evaluation {
    Centipawns(i32),
    Mate(i32),
}
use Evaluation::*;

fn parse_evaluation(line: &str) -> Option<Evaluation> {
    let parts: Vec<&str> = line.split_whitespace().collect();

    for window in parts.windows(3) {
        if window[0] == "score" {
            match window[1] {
                "cp" => {
                    let value = window[2].parse::<i32>().ok()?;
                    return Some(Centipawns(value));
                }
                "mate" => {
                    let value = window[2].parse::<i32>().ok()?;
                    return Some(Mate(value));
                }
                _ => {}
            }
        }
    }

    None
}

pub fn evaluate_fen(fen: &str, depth: u8) -> Result<Evaluation, Box<dyn std::error::Error>> {
    let mut child = Command::new("stockfish")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    let stdin = child.stdin.as_mut().unwrap();
    let stdout = child.stdout.take().unwrap();
    let reader = BufReader::new(stdout);

    // Initialize engine
    writeln!(stdin, "uci")?;
    writeln!(stdin, "isready")?;

    // Send position
    writeln!(stdin, "position fen {}", fen)?;

    // Start analysis
    writeln!(stdin, "go depth {}", depth)?;

    let mut latest_eval = None;

    for line in reader.lines() {
        let line = line?;

        if let Some(eval) = parse_evaluation(&line) {
            latest_eval = Some(eval);
        }

        if line.starts_with("bestmove") {
            break;
        }
    }

    latest_eval.ok_or_else(|| "No evaluation found".into())
}

fn example() {
    let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3";

    match evaluate_fen(fen, 12) {
        Ok(Centipawns(cp)) => {
            println!("Evaluation: {:.2}", cp as f32 / 100.0);
        }
        Ok(Mate(moves)) => {
            println!("Mate in {}", moves);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
}
