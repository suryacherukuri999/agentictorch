from agentictorch.kb.search import BM25, Chunk, Hit, best_per_url, chunk_section, fuse, tokenize


def test_tokenize_keeps_identifiers_whole_and_in_parts():
    assert tokenize("Set num_workers in torch.compile.") == [
        "set",
        "num_workers",
        "num",
        "workers",
        "in",
        "torch.compile",
        "torch",
        "compile",
    ]


def test_chunks_overlap_and_cover_the_section():
    words = [f"w{i}" for i in range(400)]
    section = {"source": "s", "heading": "h", "url": "u", "text": " ".join(words)}
    chunks = chunk_section(section, words=180, overlap=30)
    assert len(chunks) == 3
    assert chunks[1].text.split(" ")[0] == "w150"
    assert chunks[-1].text.split(" ")[-1] == "w399"
    assert chunk_section({**section, "text": "short text"})[0].text == "short text"


def test_bm25_ranks_documents_with_rare_query_terms_first():
    docs = [
        tokenize("set pin_memory and non_blocking for faster copies"),
        tokenize("the loss goes down and the model trains"),
        tokenize("the model and the loss"),
    ]
    scores = BM25(docs).scores(tokenize("pin_memory copies"))
    assert scores[0] > 0
    assert scores[1] == scores[2] == 0


def test_fuse_rewards_documents_ranked_well_by_both():
    assert fuse([1, 2, 3], [3, 1, 2])[0] == 1
    assert set(fuse([1, 2], [3])) == {1, 2, 3}


def test_keeps_the_best_chunk_of_each_section():
    def hit(url, score):
        return Hit(Chunk("s", "h", url, "t"), score)

    hits = [hit("a", 3.0), hit("b", 2.0), hit("a", 1.0)]
    assert [h.chunk.url for h in best_per_url(hits)] == ["a", "b"]
