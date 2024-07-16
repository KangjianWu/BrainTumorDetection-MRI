import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
    const [files, setFiles] = useState([]);
    const [results, setResults] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const fileInputRef = useRef(null);

    const handleFileChange = (event) => {
        setFiles(Array.from(event.target.files));
    };

    const handleUpload = async () => {
        if (files.length === 0) {
            alert("Please select at least one file.");
            return;
        }

        setIsLoading(true);
        const formData = new FormData();
        files.forEach((file) => formData.append('files', file));

        try {
            const response = await axios.post('http://127.0.0.1:5000/batch_predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            setResults(response.data.results);
        } catch (error) {
            console.error('Error uploading files:', error);
            alert('Error uploading files: ' + error.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="app-container">
            <h1>Brain Tumor MRI Segmentation</h1>

            <div className="file-upload-container">
                <h2>Upload MRI Images</h2>
                <div className="file-input-wrapper">
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileChange}
                        accept="image/*"
                        multiple
                        className="file-input"
                        style={{display: 'none'}}
                    />
                    <button onClick={() => fileInputRef.current.click()} className="select-files-button">
                        Select Files
                    </button>
                    <span>{files.length > 0 ? `${files.length} file(s) selected` : 'No files selected'}</span>
                </div>
                <button
                    onClick={handleUpload}
                    disabled={files.length === 0 || isLoading}
                    className="upload-button"
                >
                    {isLoading ? "Uploading..." : "Upload and Analyze"}
                </button>
            </div>

            {isLoading && <p>Processing images...</p>}

            {results.length > 0 && (
                <div className="results-container">
                    <h2>Results:</h2>
                    {results.map((result, index) => (
                        <div key={index} className="result-item">
                            <h3>File: {result.filename}</h3>
                            <p>Classification: {result.prediction}</p>
                            <p>Estimated Tumor Area: {result.tumor_area} pixels</p>
                            <p>Relative Tumor Size: {result.relative_size.toFixed(2)}%</p>
                            {result.mask_path && (
                                <div>
                                    <h4>Segmentation Mask:</h4>
                                    <div className="image-container">
                                        <img src={`http://127.0.0.1:5000/${result.mask_path}`} alt="Segmentation Mask" className="mask-image" />
                                        <img src={`http://127.0.0.1:5000/uploads/${result.filename}`} alt="Original MRI" className="original-image" />
                                    </div>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

export default App;
