//! KINETIC_TEST_GGUF=SmolLM2-135M-Instruct-f16.gguf KINETIC_TEST_EMBED_GGUF=bge-small-en-v1.5-f16.gguf
//! cargo test --release -p kinetic -- --include-ignored

use kinetic::{Message, Model};

fn text_model() -> Model {
    Model::open(std::env::var("KINETIC_TEST_GGUF").expect("set KINETIC_TEST_GGUF")).unwrap()
}

#[test]
fn models_cross_threads() {
    fn shared<T: Send + Sync>() {}
    shared::<Model>();
}

#[test]
fn missing_file_is_an_error() {
    let err = Model::open("no-such-model.gguf").err().unwrap();
    assert!(err.to_string().contains("no-such-model.gguf"), "{err}");
}

#[test]
#[ignore = "needs KINETIC_TEST_GGUF"]
fn greedy_chat_streams_its_answer_and_repeats() {
    let model = text_model();
    assert!(!model.is_embedding());
    let mut request = model.request();
    request.sampling.temperature = 0.0;
    request.max_tokens = Some(16);
    let messages = [Message::new("user", "What is the capital of France?")];
    let mut streamed = String::new();
    let first = model
        .chat(&messages, &request, &mut |t| {
            streamed.push_str(t);
            true
        })
        .unwrap();
    assert!(!first.content.trim().is_empty());
    assert_eq!(streamed, first.content);
    let again = model.chat(&messages, &request, &mut |_| true).unwrap();
    assert_eq!(again.content, first.content);
    assert!(model.embed(&["text"]).is_err());
}

#[test]
#[ignore = "needs KINETIC_TEST_GGUF"]
fn generation_stops_when_asked() {
    let model = text_model();
    let mut request = model.request();
    request.max_tokens = Some(64);
    let mut pieces = 0;
    let c = model
        .generate("Once upon a time", &request, &mut |_| {
            pieces += 1;
            pieces < 3
        })
        .unwrap();
    assert_eq!(pieces, 3);
    assert!(c.completion_tokens < 64);
}

#[test]
#[ignore = "needs KINETIC_TEST_GGUF"]
fn logits_are_plain_decodings_next_token_scores() {
    let model = text_model();
    let prompt = "Once upon a time";
    let logits = model.logits(prompt).unwrap();
    let n = model.n_vocab().unwrap();
    assert_eq!(logits.len(), n);
    assert!(logits.iter().all(|v| v.is_finite()));
    // Reruns give the same bits.
    assert_eq!(logits, model.logits(prompt).unwrap());
    let mut request = model.request();
    request.sampling.temperature = 0.0;
    request.max_tokens = Some(1);
    let answer = model.generate(prompt, &request, &mut |_| true).unwrap();
    // Greedy picks through the device's fused top-1 and this row comes
    // from the head read; when the top two scores sit within the rounding
    // of the two kernels the argmax and the pick can differ, so equality
    // is asserted only when the gap is decisive and the pick is one of
    // the top five always. LFM3 after this prompt, for one, has a top two
    // six thousandths of a logit apart.
    let mut top: Vec<usize> = (0..logits.len()).collect();
    top.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
    let pick = answer.output_ids[0] as usize;
    assert!(
        top[..5].contains(&pick),
        "the pick {pick} is in the top five"
    );
    if logits[top[0]] - logits[top[1]] > 0.1 {
        assert_eq!(pick, top[0], "a decisive top chooses the argmax");
    }
    // A chat prompt works too, through the model's own template.
    let messages = [Message::new("user", "What is the capital of France?")];
    assert_eq!(model.logits_chat(&messages).unwrap().len(), n);
    // An empty prompt is an error.
    assert!(model.logits("").is_err());
}

#[test]
#[ignore = "needs KINETIC_TEST_GGUF"]
fn logits_rows_score_every_position() {
    let model = text_model();
    let ids: Vec<u32> = (0..24u32).map(|i| 100 + i * 37).collect();
    let flat = model.logits_rows(&ids).unwrap();
    let n = model.n_vocab().unwrap();
    assert_eq!(flat.len(), ids.len() * n);
    assert!(flat.iter().all(|v| v.is_finite()));
    // Reruns give the same bits.
    assert_eq!(flat, model.logits_rows(&ids).unwrap());
    // The last row is what logits gives for the same tokens, where the
    // model adds no BOS of its own and the tokenizations then agree.
    if model.special_tokens().0.is_none() {
        let prompt = "Once upon a time";
        let ids = model.tokenize(prompt, true);
        assert_eq!(
            &model.logits_rows(&ids).unwrap()[ids.len() * n - n..],
            &model.logits(prompt).unwrap()[..]
        );
    }
    // A sequence scores token by token: tokens[i + 1] under row i.
    let logp = |row: &[f32], tok: u32| {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = row.iter().map(|&l| f64::from(l - max).exp()).sum();
        f64::from(row[tok as usize] - max) - sum.ln()
    };
    let score: f64 = (0..ids.len() - 1)
        .map(|i| logp(&flat[i * n..][..n], ids[i + 1]))
        .sum();
    assert!(score.is_finite() && score < 0.0);
    // An empty sequence, and a token past the vocabulary, are errors.
    assert!(model.logits_rows(&[]).is_err());
    let err = model.logits_rows(&[u32::MAX]).err().unwrap();
    assert!(err.to_string().contains("past the vocabulary"), "{err}");
}

#[test]
#[ignore = "needs KINETIC_TEST_EMBED_GGUF"]
fn an_embedding_model_has_no_logits() {
    let path = std::env::var("KINETIC_TEST_EMBED_GGUF").expect("set KINETIC_TEST_EMBED_GGUF");
    let model = Model::open(path).unwrap();
    assert!(model.n_vocab().is_none());
    let err = model.logits("a cat").err().unwrap();
    assert!(err.to_string().contains("embedding"), "{err}");
}

#[test]
#[ignore = "needs KINETIC_TEST_EMBED_GGUF"]
fn embeddings_have_the_model_width() {
    let path = std::env::var("KINETIC_TEST_EMBED_GGUF").expect("set KINETIC_TEST_EMBED_GGUF");
    let model = Model::open(path).unwrap();
    let size = model.embedding_size().unwrap();
    let v = model.embed(&["a cat", "a dog"]).unwrap();
    assert_eq!(v.len(), 2);
    assert!(v.iter().all(|e| e.len() == size));
    assert_ne!(v[0], v[1]);
    assert!(model.embed(&[""]).is_err());
}
