import io
import zipfile
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def create_test_zip(csv_content, csv_filename='test.csv'):
    """
    Utility function to create an in-memory zip file with given CSV content.
    """
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode='w') as zf:
        zf.writestr(csv_filename, csv_content)
    mem_zip.seek(0)
    return mem_zip

def test_valid_zip_with_answer(client):
    csv_content = "question,answer\nTest question,9876543210\n"
    test_zip = create_test_zip(csv_content)
    data = {
        'question': "Download and unzip file test.zip which has a single extract.csv file inside. What is the value in the 'answer' column?",
        'file': (test_zip, 'test.zip')
    }
    response = client.post("/api/", data=data, content_type='multipart/form-data')
    json_data = response.get_json()
    assert response.status_code == 200
    assert 'answer' in json_data
    assert "9876543210" in json_data['answer']

def test_zip_without_csv(client):
    # Create a zip file that does not contain any CSV file.
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode='w') as zf:
        zf.writestr("not_a_csv.txt", "Some text")
    mem_zip.seek(0)
    data = {
        'question': "Test question",
        'file': (mem_zip, 'not_a_csv.zip')
    }
    response = client.post("/api/", data=data, content_type='multipart/form-data')
    json_data = response.get_json()
    assert response.status_code == 400
    assert "error" in json_data

def test_no_file(client):
    data = {
        'question': "Test question without file"
    }
    response = client.post("/api/", data=data, content_type='multipart/form-data')
    json_data = response.get_json()
    # When no file is provided, the API uses AI Proxy for dynamic answer generation.
    assert response.status_code == 200
    assert "answer" in json_data
