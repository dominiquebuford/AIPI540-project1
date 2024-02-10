// app/page.tsx
'use client';

import { useState } from 'react';
import Image from 'next/image';
import './styles/page.css'; // Assuming you will create this CSS module

export default function Home() {
  // Changed to support an array of files
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [childName, setChildName] = useState('');
  const [responseMessage, setResponseMessage] = useState(''); // State to store response from backend
  const [isLoading, setIsLoading] = useState(false); // State to store loading status

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    // Modified to support multiple file selection
    if (event.target.files) {
      const filesArray = Array.from(event.target.files);
      setSelectedFiles(filesArray);
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
  
    if (!selectedFiles.length) {
      alert('Please select at least one file!');
      return;
    }
  
    setIsLoading(true); // Start loading
  
    const formData = new FormData();
    formData.append('childName', childName);
    selectedFiles.forEach(file => {
      formData.append('files', file);
    });
  
    try {
      const response = await fetch('http://127.0.0.1:5000/process_request', {
        method: 'POST',
        body: formData,
      });
  
      if (response.ok) {
        const data = await response.json();
        setResponseMessage(data.story); // Set the response message to display in the text box
      } else {
        alert('Failed to process request');
      }
    } catch (error) {
      console.error('There was an error processing your request:', error);
      alert('Failed to process request');
    } finally {
      setIsLoading(false); // Stop loading regardless of request outcome
    }
  };

  return (
    <div className={"container"}>
      <Image
        src="/animated-veggies.jpeg"
        alt="Animated Foods"
        width={400}
        height={160}
      />
      <h1 className={"title"}>Food Fables</h1>
      <form onSubmit={handleSubmit} className={"horizontalForm"}>
        <input
          type="text"
          placeholder="Name of the Child (Optional)"
          value={childName}
          onChange={(e) => setChildName(e.target.value)}
          className={"input"}
        />
        <input
          type="file"
          onChange={handleFileChange}
          accept="image/*"
          multiple
          className={"input-images"}
        />
        <button type="submit" className={"button"}>Submit</button>
      </form>
      {isLoading ? (
        <div className="loadingContainer">
          <div className="loader"></div>
          <div>Generating story...</div>
        </div>
      ) : (
        <textarea
          className={"responseTextbox"}
          value={responseMessage}
          readOnly
        />
      )}
    </div>
  );}
