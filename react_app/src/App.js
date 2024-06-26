// src/App.js
import React from 'react';
import FileUpload from './FileUpload';
import './App.css';

function App() {
    return (
        <div className="App">
            <header className="App-header">
                <h1>Brain Tumor MRI Segmentation</h1>
            </header>
            <main>
                <FileUpload />
            </main>
        </div>
    );
}

export default App;
