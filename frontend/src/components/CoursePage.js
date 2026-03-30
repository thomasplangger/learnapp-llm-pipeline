
import React, { useEffect, useMemo, useState, useCallback } from 'react';
import { useNavigate, useParams, Link } from 'react-router-dom';

const API_BASE = process.env.REACT_APP_BACKEND_URL || '';

function slugify(s = 'General') {
  return s
    .toString()
    .toLowerCase()
    .trim()
    .replace(/[\s\W-]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'general';
}

export default function CoursePage() {
  const { courseId } = useParams();
  const navigate = useNavigate();

  const [course, setCourse] = useState(null);
  const [lessons, setLessons] = useState([]);
  const [learningObjectives, setLearningObjectives] = useState([]);
  const [activeTab, setActiveTab] = useState('los');
  const [pdfs, setPdfs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState('');
  const [showDelete, setShowDelete] = useState(false);
  const [progressRecords, setProgressRecords] = useState([]);

  useEffect(() => {
    let active = true;
    (async () => {
      setErr('');
      setLoading(true);
      try {
        
        const res = await fetch(`${API_BASE}/api/course/${courseId}`);
        if (!res.ok) throw new Error();
        const body = await res.json();
        if (!active) return;

        setCourse(body.course);
        setLessons(body.lessons || []);

        try {
          const loRes = await fetch(`${API_BASE}/api/course/${courseId}/learning-objectives`);
          if (loRes.ok) {
            const loBody = await loRes.json();
            const groups = Array.isArray(loBody.groups) ? loBody.groups : [];
            if (active) setLearningObjectives(groups);
          } else {
            if (active) setLearningObjectives([]);
          }
        } catch {
          if (active) setLearningObjectives([]);
        }

        let coursePdfs = [];
        try {
          const pdfRes = await fetch(`${API_BASE}/api/course/${courseId}/pdfs`);
          if (pdfRes.ok) {
            coursePdfs = await pdfRes.json();
          } else {
            throw new Error();
          }
        } catch {
          const ids = body.course?.pdf_ids || [];
          if (ids.length) {
            const list = await fetch(`${API_BASE}/api/pdfs`).then(r => (r.ok ? r.json() : []));
            coursePdfs = list.filter(p => ids.includes(p.pdf_id));
          }
        }
        if (active) setPdfs(coursePdfs);
      } catch {
        if (active) setErr('Failed to load course');
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [courseId]);

  useEffect(() => {
    fetch(`${API_BASE}/api/progress/default_user`)
      .then(r => (r.ok ? r.json() : []))
      .then(setProgressRecords)
      .catch(() => {});
  }, [courseId, lessons]);

  const deleteCourse = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/course/${courseId}`, { method: 'DELETE' });
      if (res.status === 204) navigate('/');
      else throw new Error();
    } catch {
      setErr('Could not delete course');
    }
  }, [courseId, navigate]);

  const loGroups = useMemo(() => {
    if (learningObjectives && learningObjectives.length > 0) {
      return learningObjectives.map((g) => {
        const title = g.title || 'Learning Objective';
        const slug  = slugify(title);
        const losLessons = (lessons || []).filter(l => (l.section_title || 'General') === title);
        const total = losLessons.length;
        const done  = losLessons.filter(l => progressRecords.some(p => p.lesson_id === l.id && p.completed));
        const pct   = total ? Math.round((done.length / total) * 100) : 0;
        const avg   = done.length ? Math.round(done.reduce((s, l) => {
          const rec = progressRecords.find(p => p.lesson_id === l.id);
          return s + (rec?.score || 0);
        }, 0) / done.length) : 0;
        const band  = avg >= 80 ? 'green' : avg >= 50 ? 'yellow' : 'red';
        return {
          title,
          slug,
          summary: g.summary || '',
          objectives: Array.isArray(g.objectives) ? g.objectives : [],
          _stats: { total, completed: done.length, pct, avg, band }
        };
      });
    }
    const map = new Map();
    lessons.forEach(l => {
      const key = l.section_title || 'General';
      const arr = map.get(key) || [];
      arr.push(l);
      map.set(key, arr);
    });
    return Array.from(map.entries()).map(([title, list]) => {
      const total = list.length;
      const done  = list.filter(l => progressRecords.some(p => p.lesson_id === l.id && p.completed));
      const pct   = total ? Math.round((done.length / total) * 100) : 0;
      const avg   = done.length ? Math.round(done.reduce((s, l) => {
        const rec = progressRecords.find(p => p.lesson_id === l.id);
        return s + (rec?.score || 0);
      }, 0) / done.length) : 0;
      const band  = avg >= 80 ? 'green' : avg >= 50 ? 'yellow' : 'red';
      return {
        title,
        slug: slugify(title),
        summary: '',
        objectives: [],
        count: list.length,
        _stats: { total, completed: done.length, pct, avg, band }
      };
    });
  }, [learningObjectives, lessons]);

  const goToLo = slug => navigate(`/course/${courseId}/lo/${slug}`);

  if (loading) return <div className="p-8">Loading…</div>;
  if (err) return <div className="p-8 text-red-600">{err}</div>;
  if (!course) return null;

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="flex justify-between items-center mb-6">
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/')}
              className="bg-gray-200 px-4 py-2 rounded hover:bg-gray-300"
            >
              ← All Courses
            </button>
            <Link
              to={`/course/${courseId}/data`}
              className="px-4 py-2 rounded border border-gray-300 bg-white hover:bg-gray-50"
            >
              Data (Texts & Chunks)
            </Link>
          </div>

          <button
            onClick={() => setShowDelete(true)}
            className="text-red-600 px-3 py-1 rounded hover:bg-red-100"
          >
            Delete Course
          </button>
        </div>

        <h1 className="text-3xl font-bold mb-2">{course.title}</h1>
        <p className="text-gray-600 mb-6">{course.description}</p>

        <div className="flex space-x-4 border-b mb-6">
          <button
            onClick={() => setActiveTab('los')}
            className={`pb-2 ${
              activeTab === 'los' ? 'border-b-2 border-blue-600 font-semibold' : 'text-gray-600'
            }`}
          >
            Learning Objectives
          </button>
          <button
            onClick={() => setActiveTab('pdfs')}
            className={`pb-2 ${
              activeTab === 'pdfs' ? 'border-b-2 border-blue-600 font-semibold' : 'text-gray-600'
            }`}
          >
            Course PDFs
          </button>
        </div>

        {activeTab === 'los' ? (
          <div className="grid md:grid-cols-2 gap-6">
            {loGroups.length === 0 ? (
              <div className="text-gray-600">No learning objectives yet.</div>
            ) : (
              loGroups.map(lo => (
                <div key={lo.slug} className="bg-white p-6 rounded-lg shadow">
                  <h3 className="text-xl font-semibold mb-1">{lo.title}</h3>
                  {lo.summary && <p className="text-gray-700 mb-2">{lo.summary}</p>}
                  {lo._stats && (
                    <div className="mb-3">
                      <div className="flex items-center gap-3 mb-1">
                        <div className="text-sm text-gray-600">Progress</div>
                        <div className="flex-1 h-2 bg-gray-200 rounded">
                          <div
                            className={`h-2 rounded ${
                              lo._stats.band === 'green' ? 'bg-green-500' : lo._stats.band === 'yellow' ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${lo._stats.pct}%` }}
                          />
                        </div>
                        <div className="text-xs text-gray-700 w-10 text-right">{lo._stats.pct}%</div>
                      </div>
                      <div className="text-sm text-gray-600">
                        {lo._stats.completed}/{lo._stats.total} lessons · Avg score{' '}
                        <span className={`${lo._stats.band === 'green' ? 'text-green-700' : lo._stats.band === 'yellow' ? 'text-yellow-700' : 'text-red-700'} font-semibold`}>
                          {lo._stats.avg}%
                        </span>
                      </div>
                    </div>
                  )}
                  {Array.isArray(lo.objectives) && lo.objectives.length > 0 && (
                    <ul className="list-disc list-inside text-gray-700 mb-4">
                      {lo.objectives.map((o, i) => (
                        <li key={i}>{o}</li>
                      ))}
                    </ul>
                  )}
                  <button
                    onClick={() => goToLo(lo.slug)}
                    className="px-6 py-2 rounded bg-blue-600 text-white hover:bg-blue-700"
                  >
                    Start Objective
                  </button>
                </div>
              ))
            )}
          </div>
        ) : (
          <div className="grid md:grid-cols-2 gap-6">
            {pdfs.length === 0 ? (
              <p className="text-gray-600">No PDFs found for this course.</p>
            ) : (
              pdfs.map(p => (
                <div key={p.pdf_id} className="bg-white p-6 rounded-lg shadow">
                  <div className="font-semibold mb-1">{p.filename}</div>
                  <div className="text-sm text-gray-600">
                    Pages: {p.page_count} — Words: {p.word_count}
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {showDelete && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-lg shadow-lg max-w-sm w-full">
              <h2 className="text-xl font-semibold mb-4">Delete Course?</h2>
              <p className="mb-6">Permanently delete this course?</p>
              <div className="flex justify-end space-x-4">
                <button
                  onClick={() => setShowDelete(false)}
                  className="px-4 py-2 rounded bg-gray-200 hover:bg-gray-300"
                >
                  Cancel
                </button>
                <button
                  onClick={deleteCourse}
                  className="px-4 py-2 rounded bg-red-600 text-white hover:bg-red-700"
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
