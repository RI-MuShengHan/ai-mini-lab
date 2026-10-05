import json
from pathlib import Path
from contextlib import contextmanager
import uuid
import threading
import unittest
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import numpy as np
from sklearn.datasets import load_digits, load_wine
from mini_lab import digits, wine
from mini_lab.search import NoteIndex
from mini_lab.web import Handler, ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[1]
(ROOT/'.cache').mkdir(exist_ok=True)


@contextmanager
def corpus_dir():
    # A normal mkdir avoids Python 3.13's restrictive Windows temp-directory ACL.
    base=(ROOT/'.cache').resolve()
    path=base/('test-'+uuid.uuid4().hex)
    path.mkdir()
    assert path.resolve().is_relative_to(base)
    try:yield path
    finally:
        for file in path.iterdir():file.unlink()
        path.rmdir()


class DigitsTests(unittest.TestCase):
    def test_split_has_no_overlap_and_covers_data(self):
        target=load_digits().target
        a,b=digits.split_indices(target)
        self.assertFalse(set(a)&set(b))
        self.assertEqual(set(a)|set(b),set(range(len(target))))
        self.assertEqual(set(target[b]),set(range(10)))

    def test_seed_is_reproducible(self):
        y=load_digits().target
        self.assertTrue(np.array_equal(digits.split_indices(y)[1],digits.split_indices(y)[1]))

    def test_sample_prediction_and_input_validation(self):
        d=digits.sample(0)
        self.assertEqual(digits.predict_pixels(d['pixels']),d['predicted'])
        for value in [[], [[0]*8]*8, [[float('nan')]*8]*8, [[17]*8]*8]:
            with self.assertRaises(ValueError):digits.predict_pixels(value)
        with self.assertRaises(ValueError):digits.sample(-1)


class WineTests(unittest.TestCase):
    def test_scaler_is_fitted_on_training_only(self):
        model,data,a,b,_=wine.train()
        self.assertFalse(set(a)&set(b))
        np.testing.assert_allclose(model['standardscaler'].mean_,data.data[a].mean(axis=0))
        before=model['standardscaler'].mean_.copy()
        model.predict(data.data[b])
        np.testing.assert_array_equal(before,model['standardscaler'].mean_)

    def test_input_and_sample(self):
        d=wine.sample(0)
        self.assertEqual(wine.predict_features(d['values'])['predicted'],d['predicted'])
        for value in [[], [0]*13, [float('nan')]*13, [-1]*13]:
            with self.assertRaises(ValueError):wine.predict_features(value)


class SearchTests(unittest.TestCase):
    def test_retrieves_cited_lines(self):
        hit=NoteIndex(ROOT/'data/demo_notes').search('任务中断了如何避免重复生成')[0]
        self.assertEqual(hit['heading'],'Agent 中断恢复')
        raw=(ROOT/'data/demo_notes'/hit['source']).read_text(encoding='utf-8').splitlines()
        self.assertEqual('\n'.join(raw[hit['start_line']-1:hit['end_line']]),hit['text'])

    def test_empty_corpus_and_unknown_query(self):
        with corpus_dir() as p:
            self.assertEqual(NoteIndex(p).search('hello'),[])
        index=NoteIndex(ROOT/'data/demo_notes')
        self.assertEqual(index.search(''),[])
        self.assertEqual(index.search('zzzxxyyqqq'),[])

    def test_bad_topk(self):
        with self.assertRaises(ValueError):NoteIndex(ROOT/'data/demo_notes').search('test',0)

    def test_paragraph_and_heading_boundaries(self):
        with corpus_dir() as p:
            Path(p,'a.md').write_text('# One\nalpha beta\n\n## Two\ngamma delta\n',encoding='utf-8')
            hits=NoteIndex(p).search('gamma delta')
            self.assertEqual(hits[0]['heading'],'Two')
            self.assertEqual(hits[0]['start_line'],5)


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def test_page_and_health(self):
        with urlopen(self.url) as r:self.assertIn(b'AI Mini Lab',r.read())
        with urlopen(self.url+'/api/health') as r:self.assertEqual(json.load(r)['status'],'ok')

    def test_unknown_path_is_not_a_file_server(self):
        with self.assertRaises(HTTPError) as e:urlopen(self.url+'/../README.md')
        self.assertEqual(e.exception.code,404)

    def test_cross_origin_post_is_rejected(self):
        req=Request(self.url+'/api/predict-digit',data=b'{}',headers={'Origin':'https://example.com'})
        with self.assertRaises(HTTPError) as e:urlopen(req)
        self.assertEqual(e.exception.code,403)

    def test_invalid_json_is_reported(self):
        with self.assertRaises(HTTPError) as e:urlopen(Request(self.url+'/api/predict-wine',data=b'{'))
        self.assertEqual(e.exception.code,400)


if __name__=='__main__':unittest.main()
