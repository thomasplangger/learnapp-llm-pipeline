
import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

const API_BASE = process.env.REACT_APP_BACKEND_URL || '';
const MAX_LOS  = 50;

export default function FirstPage() {
  const [courses, setCourses] = useState([]);
  const [err, setErr] = useState('');

  
  const [showModal, setShowModal] = useState(false);
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [numLessons, setNumLessons] = useState('5');

  
  const [queue, setQueue] = useState([]);           
  const [uploads, setUploads] = useState([]);       
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef();

  const [creatingMode, setCreatingMode] = useState(null); 
  const [autoProgress, setAutoProgress] = useState('');
  const [autoProgressValue, setAutoProgressValue] = useState(0);
  const autoProgressIntervalRef = useRef(null);
  const autoProgressGoalRef = useRef(0);
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const navigate = useNavigate();

  const loadCourses = useCallback(async () => {
    setErr('');
    try {
      const res = await fetch(`${API_BASE}/api/courses`);
      if (!res.ok) throw new Error();
      setCourses(await res.json());
    } catch {
      setErr('Failed to load courses');
    }
  }, []);

  useEffect(() => { loadCourses(); }, [loadCourses]);

  const openCreate = () => {
    setTitle(''); setDescription(''); setNumLessons('5');
    setQueue([]); setUploads([]); setErr('');
    resetAutoProgress();
    setCreatingMode(null);
    setShowModal(true);
  };

  const onDrop = e => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files || []).filter(f => f.type === 'application/pdf');
    if (files.length) setQueue(prev => [...prev, ...files]);
  };
  const onDragOver = e => e.preventDefault();

  const onPick = e => {
    const files = Array.from(e.target.files || []).filter(f => f.type === 'application/pdf');
    if (files.length) setQueue(prev => [...prev, ...files]);
    e.target.value = '';
  };

  const uploadOne = async (file) => {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch(`${API_BASE}/api/pdf/upload`, { method: 'POST', body: fd });
    if (!r.ok) throw new Error('upload failed');
    return r.json();
  };

  const processQueue = useCallback(async () => {
    if (uploading || queue.length === 0) return;
    setUploading(true);
    try {
      for (const f of queue) {
        try {
          const info = await uploadOne(f);
          setUploads(prev => [...prev, info]);
        } catch {
          setErr(prev => prev ? prev : `Upload failed for ${f.name}`);
        }
      }
      setQueue([]);
    } finally {
      setUploading(false);
    }
  }, [queue, uploading]);

  useEffect(() => { processQueue(); }, [queue, processQueue]);

  const removeUpload = async (pdf_id) => {
    try {
      const res = await fetch(`${API_BASE}/api/pdf/${pdf_id}`, { method: 'DELETE' });
      if (res.status !== 204) throw new Error();
      setUploads(prev => prev.filter(u => u.pdf_id !== pdf_id));
    } catch {
      setErr('Could not delete PDF');
    }
  };

  const renameUpload = async (pdf_id, newName) => {
    if (!newName.trim()) return;
    try {
      const res = await fetch(`${API_BASE}/api/pdf/${pdf_id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: newName.trim() })
      });
      if (!res.ok) throw new Error();
      setUploads(prev => prev.map(u => u.pdf_id === pdf_id ? { ...u, filename: newName.trim() } : u));
    } catch {
      setErr('Rename failed');
    }
  };

  // --- Auto-create progress helpers ---
  const startAutoProgress = (label = 'Initializing course…', goal = 20) => {
    setAutoProgress(label);
    setAutoProgressValue(5);
    autoProgressGoalRef.current = goal;
    if (autoProgressIntervalRef.current) {
      clearInterval(autoProgressIntervalRef.current);
    }
    autoProgressIntervalRef.current = setInterval(() => {
      setAutoProgressValue(prev => {
        const target = autoProgressGoalRef.current || 0;
        if (prev >= target) return prev;
        const delta = target - prev;
        const step = Math.min(1.1, Math.max(0.18, delta * 0.035));
        return Math.min(target, prev + step);
      });
    }, 220);
  };

  const advanceAutoProgress = (label, goal) => {
    setAutoProgress(label);
    autoProgressGoalRef.current = goal;
  };

  const completeAutoProgress = (label = 'Finalizing course…') => {
    advanceAutoProgress(label, 100);
  };

  const resetAutoProgress = () => {
    if (autoProgressIntervalRef.current) {
      clearInterval(autoProgressIntervalRef.current);
      autoProgressIntervalRef.current = null;
    }
    autoProgressGoalRef.current = 0;
    setAutoProgress('');
    setAutoProgressValue(0);
  };

  useEffect(() => {
    return () => {
      if (autoProgressIntervalRef.current) {
        clearInterval(autoProgressIntervalRef.current);
      }
    };
  }, []);

  const requestLearningObjectivesForCourse = async (courseId, payload) => {
    const res = await fetch(`${API_BASE}/api/course/${courseId}/learning-objectives`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const msg = await res.text();
      throw new Error(`Learning objective generation failed: ${msg || res.status}`);
    }
    return res.json();
  };

  const verifyLearningObjectivesPersisted = async (courseId, retries = 3) => {
    for (let attempt = 0; attempt < retries; attempt += 1) {
      try {
        const res = await fetch(`${API_BASE}/api/course/${courseId}/learning-objectives`);
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data?.groups) && data.groups.length > 0) {
            return true;
          }
        }
      } catch {
      }
      await sleep(600 + attempt * 250);
    }
    return false;
  };

  const ensureLearningObjectivesReady = async (courseId, loCount) => {
    const basePayload = { persist: true };
    if (Number.isFinite(loCount) && loCount > 0) {
      basePayload.N = loCount;
    }
    const attempts = [
      { method: 'llm', label: 'Generating learning objectives…', goal: 90 },
      { method: 'heuristic', label: 'Generating learning objectives… (heuristic fallback)', goal: 93 },
    ];
    for (const attempt of attempts) {
      advanceAutoProgress(attempt.label, attempt.goal);
      await requestLearningObjectivesForCourse(courseId, { ...basePayload, method: attempt.method });
      if (await verifyLearningObjectivesPersisted(courseId)) {
        return;
      }
    }
    throw new Error('Learning objectives are not ready yet. Please use Debug Mode to finish setup.');
  };

  const runAutoPipeline = async (courseId, loCount) => {
    advanceAutoProgress('Chunking course material…', 40);
    const chunkPayload = {
      method: 'LLM [JSON]',
      overwrite: true,
      dry_run: false,
    };
    const chunkRes = await fetch(`${API_BASE}/api/course/${courseId}/chunk`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(chunkPayload),
    });
    if (!chunkRes.ok) {
      const msg = await chunkRes.text();
      throw new Error(`Chunking failed: ${msg || chunkRes.status}`);
    }

    advanceAutoProgress('Refreshing chunk metadata…', 65);
    const refreshRes = await fetch(`${API_BASE}/api/course/${courseId}/chunks/refresh-metadata`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ persist: true }),
    });
    if (!refreshRes.ok) {
      const msg = await refreshRes.text();
      throw new Error(`Chunk metadata refresh failed: ${msg || refreshRes.status}`);
    }

    await ensureLearningObjectivesReady(courseId, loCount);
  };

  const createCourse = async (mode) => {
    if (!uploads.length) { setErr('Please upload at least one PDF'); return; }
    setCreatingMode(mode);
    setErr('');
    resetAutoProgress();
    if (mode === 'auto') {
      startAutoProgress('Initializing course…', 25);
    }

    const parsedLOs = parseInt(numLessons, 10);
    const normalizedLOs = Number.isFinite(parsedLOs) && parsedLOs > 0 ? parsedLOs : null;
    let newCourseId = null;

    try {
      const payload = {
        title: title.trim() || null,
        description: description.trim() || null,
        pdf_ids: uploads.map(u => u.pdf_id),
      };
      if (normalizedLOs) {
        payload.num_lessons = normalizedLOs;
      }

      const res = await fetch(`${API_BASE}/api/create-course`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error('Failed to create course');
      const data = await res.json();
      newCourseId = data.course_id;
      if (!newCourseId) throw new Error('Missing course ID in response');

      if (mode === 'debug') {
        setShowModal(false);
        navigate(`/course/${newCourseId}/data?mode=debug`);
        return;
      }

      await runAutoPipeline(newCourseId, normalizedLOs);
      completeAutoProgress();
      await new Promise(resolve => setTimeout(resolve, 450));
      setShowModal(false);
      const suffix = normalizedLOs ? `?mode=auto&autoLos=${normalizedLOs}` : '?mode=auto';
      navigate(`/course/${newCourseId}/data${suffix}`);
    } catch (error) {
      console.error(error);
      setErr(error?.message || 'Failed to create course');
      if (mode === 'auto' && newCourseId) {
        alert('Automatic setup failed. Opening Debug Mode so you can finish manually.');
        setShowModal(false);
        navigate(`/course/${newCourseId}/data?mode=debug`);
      }
    } finally {
      setCreatingMode(null);
      if (mode === 'auto') {
        resetAutoProgress();
      } else {
        setAutoProgress('');
      }
    }
  };

  const deleteCourse = async (courseId) => {
    if (!window.confirm('Delete this course?')) return;
    try {
      const res = await fetch(`${API_BASE}/api/course/${courseId}`, { method: 'DELETE' });
      if (res.status !== 204) throw new Error();
      loadCourses();
    } catch {
      setErr('Could not delete course');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">

        <div className="flex items-center justify-between mb-8">
          <h1 className="text-4xl font-bold">Courses</h1>
          <div className="space-x-2">
            <button
              onClick={() => navigate('/pdfs')}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              My PDFs
            </button>
            <button
              onClick={() => navigate('/autotest')}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Auto Test
            </button>
            <button
              onClick={openCreate}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Create Course
            </button>
          </div>
        </div>

        {courses.length === 0 ? (
          <p className="text-gray-600">No courses yet. Create your first one!</p>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {courses.map(c => (
              <div key={c.id} className="relative bg-white p-6 rounded-lg shadow hover:shadow-lg">
                <button
                  onClick={() => deleteCourse(c.id)}
                  className="absolute top-2 right-2 text-gray-800 text-xl"
                  title="Delete course"
                >
                  ✕
                </button>
                <h3 className="text-xl font-semibold mb-2">{c.title}</h3>
                <p className="text-gray-600 mb-4">{c.description}</p>
                <div className="text-sm text-gray-600 mb-2">
                  Learning Objectives: {c.lessons?.length ?? 0}
                </div>
                <button
                  onClick={() => navigate(`/course/${c.id}`)}
                  className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700"
                >
                  Open
                </button>
              </div>
            ))}
          </div>
        )}

        {err && <div className="mt-6 max-w-xl bg-red-100 text-red-700 p-4 rounded">{err}</div>}
      </div>

      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
          <div className="bg-white w-full max-w-2xl p-6 rounded-lg shadow">
            <h2 className="text-2xl font-bold mb-4">Create Course</h2>

            <label className="block text-sm mb-1">Course title</label>
            <input
              className="w-full border rounded p-2 mb-3"
              value={title}
              onChange={e => setTitle(e.target.value)}
              placeholder="e.g., Linear Algebra for CS"
            />
            <label className="block text-sm mb-1">Description</label>
            <textarea
              className="w-full border rounded p-2 mb-3"
              rows={3}
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder="Short description…"
            />
            <label className="block text-sm mb-1">Number of Learning Objectives</label>
            <input
              type="number"
              min="1"
              max={MAX_LOS}
              className="w-28 border rounded p-2 mb-4"
              value={numLessons}
              onChange={e => setNumLessons(e.target.value)}
            />

            <div className="mb-4">
              <div
                onDrop={onDrop}
                onDragOver={onDragOver}
                onClick={() => fileInputRef.current.click()}
                className="border-2 border-dashed border-gray-400 rounded-lg p-8 text-center bg-gray-50 hover:bg-gray-100 cursor-pointer"
              >
                <p className="text-gray-600">
                  Drag & drop PDF files here, or click to browse
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  multiple
                  onChange={onPick}
                  className="hidden"
                />
              </div>

              {queue.length > 0 && (
                <div className="mt-4 bg-yellow-50 border border-yellow-200 rounded p-4">
                  <div className="font-semibold mb-2">Pending uploads</div>
                  <ul className="space-y-2">
                    {queue.map((f, i) => (
                      <li key={i} className="flex justify-between items-center">
                        <span className="text-sm text-gray-800">{f.name}</span>
                        <span className="text-xs text-gray-500">{(f.size/1024/1024).toFixed(2)} MB</span>
                      </li>
                    ))}
                  </ul>
                  <div className="text-sm text-gray-600 mt-2">
                    {uploading ? 'Uploading…' : 'Uploading will start automatically'}
                  </div>
                </div>
              )}

              <div className="mt-4">
                <div className="font-semibold mb-2">Uploaded files</div>
                {uploads.length === 0 ? (
                  <div className="text-gray-500 text-sm">No files uploaded yet.</div>
                ) : (
                  <div className="space-y-3">
                    {uploads.map(u => (
                      <div key={u.pdf_id} className="bg-gray-50 p-3 rounded border">
                        <div className="flex items-center justify-between">
                          <input
                            className="flex-1 border-b p-1 mr-2"
                            value={u.filename}
                            onChange={e =>
                              setUploads(prev =>
                                prev.map(x => x.pdf_id === u.pdf_id ? { ...x, filename: e.target.value } : x)
                              )
                            }
                          />
                          <div className="text-xs text-gray-600 mr-2">
                            {u.page_count} pages · {u.word_count} words
                          </div>
                          <button
                            onClick={() => renameUpload(u.pdf_id, u.filename)}
                            className="px-2 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 mr-2"
                          >
                            Rename
                          </button>
                          <button
                            onClick={() => removeUpload(u.pdf_id)}
                            className="px-2 py-1 text-sm bg-red-600 text-white rounded hover:bg-red-700"
                          >
                            Remove
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="flex justify-end space-x-2">
              <button
                onClick={() => setShowModal(false)}
                disabled={creatingMode !== null}
                className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={() => createCourse('debug')}
                disabled={creatingMode !== null || uploads.length === 0}
                className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 disabled:bg-gray-400"
              >
                {creatingMode === 'debug' ? 'Opening Debug…' : 'Debug Mode'}
              </button>
              <button
                onClick={() => createCourse('auto')}
                disabled={creatingMode !== null || uploads.length === 0}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-400"
              >
                {creatingMode === 'auto' ? 'Creating…' : 'Create Course'}
              </button>
            </div>

            {creatingMode === 'auto' && (
              <div className="mt-4 p-4 border border-blue-200 rounded-lg bg-blue-50">
                <div className="text-base font-semibold text-blue-900 mb-2">
                  {autoProgress || 'Preparing course…'}
                </div>
                <div className="w-full h-3 bg-blue-100 rounded-full overflow-hidden">
                  <div
                    className="h-3 bg-blue-600 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, autoProgressValue).toFixed(1)}%` }}
                  />
                </div>
                <div className="text-xs text-blue-800 mt-1">
                  {Math.min(100, Math.round(autoProgressValue))}%
                </div>
              </div>
            )}

            {err && <div className="mt-3 bg-red-100 text-red-700 p-3 rounded">{err}</div>}
          </div>
        </div>
      )}
    </div>
  );
}
