from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_health():
    r = client.get('/api/health')
    assert r.status_code == 200 and r.json()['documents'] == 4

def test_inverter_citation():
    d = client.post('/api/run', json={'text': 'What should be checked for inverter derating?'}).json()
    assert 'solar_inverter_sop.md' in d['sources']
    assert 'cooling fans' in d['answer']

def test_hydro_safety_citation():
    d = client.post('/api/run', json={'text': 'What checks are required before spillway gate inspection?'}).json()
    assert d['sources'] == ['hydro_safety_brief.md']
    assert 'lockout-tagout' in d['answer']

def test_unsupported_question_abstains():
    d = client.post('/api/run', json={'text': 'What is the capital of France?'}).json()
    assert d['abstained'] is True
    assert d['sources'] == []
