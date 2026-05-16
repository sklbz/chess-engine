use std::error::Error;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

#[derive(Debug, Copy, Clone)]
pub enum Evaluation {
    Centipawns(i32),
    Mate(i32),
}

use Evaluation::*;

pub struct StockfishProcess {
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
}

impl StockfishProcess {
    pub fn spawn() -> Result<StockfishProcess, Box<dyn Error>> {
        let mut child = Command::new("stockfish")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;

        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let reader = BufReader::new(stdout);

        Ok(StockfishProcess { stdin, reader })
    }

    pub fn evaluate_fen(
        &mut self,
        fen: &str,
        depth: u8,
    ) -> Result<Evaluation, Box<dyn std::error::Error>> {
        writeln!(self.stdin, "position fen {}", fen)?;
        writeln!(self.stdin, "go depth {}", depth)?;
        let mut latest_eval = None;

        for line in self.reader.by_ref().lines() {
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
}

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
