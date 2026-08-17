from hotloop.kb.fetch import sections

PYTORCH_PAGE = """
<html><body><nav>Table of contents that should be skipped</nav>
<main><article class="bd-article">
<section id="cuda-semantics">
<h1>CUDA semantics<a class="headerlink" href="#cuda-semantics">#</a></h1>
<p>torch.cuda keeps track of the currently selected GPU and runs work asynchronously.</p>
<section id="asynchronous-execution">
<h2>Asynchronous execution<a class="headerlink" href="#asynchronous-execution">#</a></h2>
<p>By default, GPU operations are asynchronous, so the CPU can queue more work.</p>
<nav>On this page: skipped</nav>
</section></section>
</article></main></body></html>
"""

NVIDIA_PAGE = """
<html><body><main class="Page-main"><form>Search all documentation</form>
<div class="RichTextBody"><h2><a class="Link" href="#abstract" id="abstract">Abstract</a></h2>
<p>Mixed precision methods combine the use of different numerical formats in one workload.</p>
</div></main></body></html>
"""


def test_splits_a_pytorch_page_into_sections_with_anchors():
    found = sections("CUDA semantics", "https://docs/cuda.html", PYTORCH_PAGE)
    assert [s.heading for s in found] == ["CUDA semantics", "Asynchronous execution"]
    assert found[1].url == "https://docs/cuda.html#asynchronous-execution"
    assert "CPU can queue more work" in found[1].text
    assert "skipped" not in " ".join(s.text for s in found)


def test_reads_nvidia_pages_without_an_article():
    [section] = sections("NVIDIA", "https://nv/mp.html", NVIDIA_PAGE)
    assert section.heading == "Abstract"
    assert section.url == "https://nv/mp.html#abstract"
    assert "Search all documentation" not in section.text
