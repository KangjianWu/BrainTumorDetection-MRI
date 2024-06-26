import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

function FileUpload() {
    const [file, setFile] = useState(null);
    const [classification, setClassification] = useState('');
    const [maskPath, setMaskPath] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const fileInputRef = useRef(null);

    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        setFile(selectedFile);
    };

    const handleUpload = async () => {
        if (!file) {
            alert("Please select a file first.");
            return;
        }

        setIsLoading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            setClassification(response.data.prediction);
            setMaskPath(response.data.mask_path);
        } catch (error) {
            console.error('Error uploading file:', error);
            alert('Error uploading file: ' + error.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="file-upload-container">
            <h2>Upload MRI Image</h2>
            <div className="file-input-wrapper">
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept="image/*"
                    className="file-input"
                />
                <button
                    onClick={() => fileInputRef.current.click()}
                    className="custom-file-button"
                >
                    {file ? file.name : "Choose a file"}
                </button>
            </div>
            <button onClick={handleUpload} disabled={!file || isLoading} className="upload-button">
                {isLoading ? "Uploading..." : "Upload"}
            </button>

            {isLoading && <p>Processing image...</p>}

            {classification && (
                <div className="result-container">
                    <h3>Classification: {classification}</h3>
                    {maskPath && (
                        <div>
                            <h3>Segmentation Mask:</h3>
                            <img src={`http://127.0.0.1:5000/${maskPath}`} alt="Segmentation Mask" className="mask-image" />
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default FileUpload;