import axios from 'axios';  // npm install axios

const UploadComponent = () => {
  const handleSubmit = async (files, jobDesc) => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    formData.append('job_desc', jobDesc);

    try {
      const response = await axios.post('http://localhost:8000/upload-resumes', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      // Update UI: setRankings(response.data.rankings); setSwaps(response.data.swaps);
    } catch (error) { console.error('API Error:', error); }
  };
  return <input type="file" multiple onChange={...} />;  // Your upload form
};
