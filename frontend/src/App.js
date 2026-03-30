
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import FirstPage from './components/FirstPage';
import CoursePage from './components/CoursePage';
import LearningObjectivePage from './components/LearningObjectivePage';
import LessonView from './components/LessonView';
import PdfLibraryView from './components/PdfLibraryView';
import CourseDataPage from "./components/CourseDataPage";
import AutoTestPage from './components/AutoTestPage';

import './App.css';

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<FirstPage />} />
        <Route path="/course/:courseId" element={<CoursePage />} />
        <Route path="/course/:courseId/lo/:loSlug" element={<LearningObjectivePage />} />
        <Route path="/course/:courseId/lesson/:lessonId" element={<LessonView />} />
        <Route path="/pdfs" element={<PdfLibraryView />} />
        <Route path="*" element={<FirstPage />} />
        <Route path="/course/:courseId/data" element={<CourseDataPage />} />
        <Route path="/autotest" element={<AutoTestPage />} />
      </Routes>
    </Router>
  );
}
