use trainer::game::{Card, Game, Hand, Play, Rank, CARDS_PER_HAND};
use trainer::net_interface::*;
use trainer::neuralnet::NeuralNet;

fn run_net_through(model: &mut NeuralNet, hand: &Hand, faceup: &Card) -> Play {
    println!("{}", hand);
    println!("Faceup {}", faceup);
    let mut input_buf = [0.0; INPUT_SIZE];
    encode_inputs(hand, faceup, &mut input_buf);
    let output = model.infer(&input_buf);
    let ret = decode_outputs(output, CARDS_PER_HAND);
    println!("Played {:?}", ret);
    ret
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut game = Game::new(2);
    let mut player_a = NeuralNet::load("golf.bc")?;
    let mut player_b = player_a.clone();
    for _ in 0..18 {
        println!("\u{1b}[93mPlayer A");
        game.play(0, |hand, faceup| run_net_through(&mut player_a, hand, faceup));
        println!("score: {}", game.players[0].score());

        println!("\u{1b}[39mPlayer B");
        game.play(1, |hand, faceup| run_net_through(&mut player_b, hand, faceup));
        println!("score: {}", game.players[1].score());
    }
    Ok(())
}
