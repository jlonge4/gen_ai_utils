use std::collections::HashMap;

struct BasicTokenizer {
    merges: HashMap<(i32, i32), i32>,
    vocab: HashMap<i32, Vec<u8>>,
}

impl BasicTokenizer {
    fn get_stats(ids: &[i32]) -> HashMap<(i32, i32), i32> {
        let mut counts = HashMap::new();
    
        for pair in ids.windows(2) {
            let count = counts.entry((pair[0], pair[1])).or_insert(0);
            *count += 1;
        }
    
        counts
    }

    fn merge(&self, ids: &[i32], pair: &(i32, i32), idx: i32) -> Vec<i32> {
        let mut newids = Vec::new();
        let mut i = 0;

        while i < ids.len() {
            if ids[i] == pair.0 && i < (ids.len()-1) && ids[i+1] == pair.1 {
                newids.push(idx);
                i += 2;
            } else {
                newids.push(ids[i]);
                i += 1;
            }
        }
        newids
    }


    fn train(&mut self, text: &str, vocab_size: usize, verbose: bool) {
        assert!(vocab_size >= 256);
        let num_merges = vocab_size - 256;
    
        // Input text preprocessing
        let text_bytes = text.as_bytes(); // raw bytes
        let mut ids: Vec<i32> = text_bytes.iter().map(|&b| b as i32).collect(); // list of integers
    
        // Iteratively merge the most common pairs to create new tokens
        let mut merges = HashMap::new(); // (i32, i32) -> i32
        let mut vocab = HashMap::new(); // i32 -> Vec<u8>
        for idx in 0..256 {
            vocab.insert(idx as i32, vec![idx as u8]);
        }
        for i in 0..num_merges {
            // Count up the number of times every consecutive pair appears
            let stats = BasicTokenizer::get_stats(&ids);
    
            // Find the pair with the highest count
            let mut max_pair = None;
            let mut max_count = 0;
            for (&pair, &count) in &stats {
                if count > max_count {
                    max_pair = Some(pair);
                    max_count = count;
                }
            }
    
            // Handle the case when stats is empty
            let pair = match max_pair {
                Some(pair) => pair,
                None => break, // Exit the loop if stats is empty
            };
    
            // Mint a new token: assign it the next available id
            let idx = 256 + i as i32;
    
            // Replace all occurrences of pair in ids with idx
            ids = self.merge(&ids, &pair, idx);
    
            // Save the merge
            merges.insert(pair, idx);
    
            // Concatenate bytes in vocab to create new token
            let token = vocab[&pair.0].iter().cloned().chain(vocab[&pair.1].iter().cloned()).collect();
            vocab.insert(idx, token);
    
            // Prints
            if verbose {
                println!(
                    "merge {}/{}: {:?} -> {} ({:?}) had {} occurrences",
                    i + 1,
                    num_merges,
                    pair,
                    idx,
                    vocab[&idx],
                    stats.get(&pair).cloned().unwrap_or_default()
                );
            }
        }
    
        // Save class variables
        self.merges = merges; // used in encode()
        self.vocab = vocab;   // used in decode()
    }


    fn decode(&self, ids: &[i32]) -> String {
        let text_bytes: Vec<u8> = ids.iter().map(|&idx| self.vocab[&idx].clone()).flatten().collect();
        let text = String::from_utf8_lossy(&text_bytes).to_string();
        text
    }

    fn encode(&self, text: &str) -> Vec<i32> {
        let text_bytes = text.as_bytes(); // raw bytes
        let mut ids: Vec<i32> = text_bytes.iter().map(|&b| b as i32).collect(); // list of integers
    
        while ids.len() >= 2 {
            // Find the pair with the lowest merge index
            let mut min_pair = None;
            let mut min_merge_index = i32::MAX;
            let stats = BasicTokenizer::get_stats(&ids);
            for (&pair, &count) in &stats {
                if let Some(&merge_index) = self.merges.get(&pair) {
                    if merge_index < min_merge_index {
                        min_pair = Some(pair);
                        min_merge_index = merge_index;
                    }
                }
            }
    
            // Check if there are no more merges available
            let pair = match min_pair {
                Some(pair) => pair,
                None => break, // nothing else can be merged anymore
            };
    
            // Merge the best pair (lowest merge index)
            let idx = self.merges[&pair];
            ids = self.merge(&ids, &pair, idx);
        }
    
        ids
    }
    
}

fn main() {
    let mut tokenizer = BasicTokenizer {
        merges: HashMap::new(),
        vocab: HashMap::new(),
    };
    let text = "DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference";
    tokenizer.train(text, 300, true);
    print!("T encoded is: {:?}", tokenizer.encode("T"));
    print!("The ID 84 decoded is: {:?}", tokenizer.decode(&[84]));

}

