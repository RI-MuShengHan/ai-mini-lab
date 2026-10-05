"""Character TF-IDF retrieval with exact source/heading/line references."""
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    source: str
    heading: str
    start_line: int
    end_line: int
    text: str


def read_chunks(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Corpus directory does not exist.')
    chunks = []
    for path in sorted(root.rglob('*.md')):
        if not path.resolve().is_relative_to(root):
            continue  # Do not follow a file symlink outside the chosen corpus.
        heading, lines, start = path.stem, [], 0
        def flush():
            if lines:
                chunks.append(Chunk(path.relative_to(root).as_posix(), heading, start,
                                    start + len(lines) - 1, '\n'.join(lines)))
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if re.match(r'^#{1,6}\s', line):
                flush(); lines = []; heading = line.lstrip('# ').strip()
            elif not line.strip():
                flush(); lines = []
            else:
                if not lines: start = number
                lines.append(line)
                if len('\n'.join(lines)) >= 900:
                    flush(); lines = []
        flush()
    return chunks


class NoteIndex:
    def __init__(self, root):
        self.chunks = read_chunks(root)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), sublinear_tf=True)
        self.matrix = None
        if self.chunks:
            try:
                self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])
            except ValueError:
                pass  # Empty vocabulary: searchable input needs at least two characters.

    def search(self, query, top_k=3):
        if not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError('top_k must be between 1 and 20.')
        if not query.strip() or self.matrix is None:
            return []
        vector = self.vectorizer.transform([query])
        if not vector.nnz:
            return []
        scores = cosine_similarity(vector, self.matrix).ravel()
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [{**asdict(self.chunks[i]), 'score': round(float(scores[i]), 4)}
                for i in order[:top_k] if scores[i] > 0]
