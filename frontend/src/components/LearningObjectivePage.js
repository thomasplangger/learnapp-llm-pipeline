
import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

const API_BASE = process.env.REACT_APP_BACKEND_URL || '';
const USER_ID  = 'default_user';

function slugify(s = 'General') {
  return s.toString().toLowerCase().trim()
    .replace(/[\s\W-]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'general';
}

export default function LearningObjectivePage() {
  const { courseId, loSlug } = useParams();
  const navigate = useNavigate();

  const [course, setCourse] = useState(null);
  const [allLessons, setAllLessons] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState('');

  const [progress, setProgress] = useState([]);
  
  const [pdfs, setPdfs] = useState([]);
  const [chunks, setChunks] = useState([]);
  const [loGroups, setLoGroups] = useState([]);
  const [showSources, setShowSources] = useState(false);
  const [activePdfId, setActivePdfId] = useState(null);

  
  useEffect(() => {
    let active = true;
    (async () => {
      setErr(''); setLoading(true);
      try {
        const res = await fetch(`${API_BASE}/api/course/${courseId}`);
        if (!res.ok) throw new Error();
        const body = await res.json();
        if (!active) return;
        setCourse(body.course);
        setAllLessons(body.lessons || []);
        try {
          const pdfRes = await fetch(`${API_BASE}/api/course/${courseId}/pdfs`);
          if (pdfRes.ok) {
            const list = await pdfRes.json();
            if (active) setPdfs(Array.isArray(list) ? list : []);
          }
        } catch {}
        try {
          const loRes = await fetch(`${API_BASE}/api/course/${courseId}/learning-objectives`);
          if (loRes.ok) {
            const data = await loRes.json();
            if (active) setLoGroups(Array.isArray(data.groups) ? data.groups : []);
          }
        } catch {}
        try {
          const chRes = await fetch(`${API_BASE}/api/chunks?course_id=${courseId}`);
          if (chRes.ok) {
            const rows = await chRes.json();
            if (active) setChunks(Array.isArray(rows) ? rows : []);
          }
        } catch {}
      } catch {
        if (active) setErr('Failed to load learning objective');
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => { active = false };
  }, [courseId]);

  useEffect(() => {
    fetch(`${API_BASE}/api/progress/${USER_ID}`)
      .then(r => (r.ok ? r.json() : []))
      .then(setProgress)
      .catch(() => {});
  }, [courseId]);

  const lo = useMemo(() => {
    const bySection = new Map();
    allLessons.forEach(l => {
      const key = l.section_title || 'General';
      const arr = bySection.get(key) || [];
      arr.push(l);
      bySection.set(key, arr);
    });
    for (const [title, lessons] of bySection.entries()) {
      if (slugify(title) === loSlug) {
        return { title, lessons };
      }
    }
    return { title: 'Learning Objective', lessons: [] };
  }, [allLessons, loSlug]);

  const loChunkIndices = useMemo(() => {
    if (!Array.isArray(loGroups) || loGroups.length === 0) return [];
    const entry = loGroups.find(g => slugify(g.title || '') === loSlug);
    return Array.isArray(entry?.chunk_indices) ? entry.chunk_indices.map(n => Number(n)) : [];
  }, [loGroups, loSlug]);

  const sources = useMemo(() => {
    if (!loChunkIndices.length || !Array.isArray(chunks)) return [];
    const byId = new Map();
    loChunkIndices.forEach((idx) => {
      const c = chunks[idx];
      const sid = c?.meta?.source_pdf_id;
      const sidx = c?.meta?.source_pdf_index;
      if (sid == null && sidx == null) return;
      const item = byId.get(sid || `idx:${sidx}`) || { pdf_id: sid || null, pdf_index: sidx, chunk_indices: [] };
      item.chunk_indices.push(idx);
      byId.set(sid || `idx:${sidx}`, item);
    });
    const nameMap = new Map((pdfs || []).map(p => [p.pdf_id, p.filename]));
    return Array.from(byId.values()).map(x => ({
      pdf_id: x.pdf_id,
      pdf_index: typeof x.pdf_index === 'number' ? x.pdf_index : null,
      filename: (x.pdf_id && nameMap.get(x.pdf_id)) || (typeof x.pdf_index === 'number' ? `PDF #${x.pdf_index + 1}` : 'Unknown PDF'),
      chunk_indices: x.chunk_indices.sort((a,b)=>a-b),
    })).sort((a,b) => (a.pdf_index ?? 9999) - (b.pdf_index ?? 9999));
  }, [loChunkIndices, chunks, pdfs]);

  useEffect(() => {
    let timer;
    let cancelled = false;
    const pending = lo.lessons.some(l => !Array.isArray(l.questions) || l.questions.length === 0);
    if (pending) {
      const poll = async () => {
        try {
          const res = await fetch(`${API_BASE}/api/course/${courseId}`);
          if (!res.ok) return;
          const body = await res.json();
          if (cancelled) return;
          setCourse(body.course);
          setAllLessons(body.lessons || []);
          const still = (body.lessons || []).some(l => !Array.isArray(l.questions) || l.questions.length === 0);
          if (still) timer = setTimeout(poll, 2000);
        } catch {
          timer = setTimeout(poll, 4000);
        }
      };
      timer = setTimeout(poll, 1500);
    }
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [courseId, lo.lessons]);

  const lessons = useMemo(() => {
    const idsDone = new Set(progress.filter(p => p.completed).map(p => p.lesson_id));
    return lo.lessons.map(l => {
      const hasDetail = Array.isArray(l.questions) && l.questions.length;
      const status = idsDone.has(l.id) ? 'done' : (hasDetail ? 'ready' : 'pending');
      const prog = progress.find(p => p.lesson_id === l.id);
      return { ...l, status, score: prog?.score ?? null };
    });
  }, [lo.lessons, progress]);

  const loStats = useMemo(() => {
    const total = lessons.length || 0;
    const done  = lessons.filter(l => l.status === 'done');
    const pct   = total ? Math.round((done.length / total) * 100) : 0;
    const avg   = done.length ? Math.round(done.reduce((s, l) => s + (l.score ?? 0), 0) / done.length) : 0;
    const band  = avg >= 80 ? 'green' : avg >= 50 ? 'yellow' : 'red';
    return { total, completed: done.length, pct, avg, band };
  }, [lessons]);

  const startLesson = (lesson) => {
    if (lesson.status === 'pending') return;
    navigate(`/course/${courseId}/lesson/${lesson.id}`);
  };

  if (loading) return <div className="p-8">Loading…</div>;
  if (err)     return <div className="p-8 text-red-600">{err}</div>;

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="mx-auto px-4 py-8 max-w-none">
        <div className="flex justify-between items-center mb-6">
          <button
            onClick={() => navigate(`/course/${courseId}`)}
            className="bg-gray-200 px-4 py-2 rounded hover:bg-gray-300"
          >
            ← {course?.title || 'Course'}
          </button>
        </div>
        <div className={`${showSources ? '' : 'w-full md:w-1/2 mx-auto'}`}>
          <h1 className="text-3xl font-bold mb-2">{lo.title}</h1>
          <div className="mb-6">
            <div className="flex items-center gap-4 mb-2">
              <div className="text-gray-700 font-medium">Progress</div>
              <div className="flex-1 h-3 bg-gray-200 rounded">
                <div
                  className={`h-3 rounded ${
                    loStats.band === 'green' ? 'bg-green-500' : loStats.band === 'yellow' ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${loStats.pct}%` }}
                />
              </div>
              <div className="text-sm text-gray-700 w-14 text-right">{loStats.pct}%</div>
            </div>
            <div className="text-gray-600">
              {loStats.completed}/{loStats.total} lessons completed · Avg score{' '}
              <span className={`${loStats.band === 'green' ? 'text-green-700' : loStats.band === 'yellow' ? 'text-yellow-700' : 'text-red-700'} font-semibold`}>
                {loStats.avg}%
              </span>
            </div>
          </div>
        </div>

        <div className="mb-4 flex justify-end">
          <button
            onClick={() => setShowSources(s => !s)}
            className="px-4 py-2 rounded border border-gray-300 bg-white hover:bg-gray-50"
          >
            {showSources ? 'Hide Sources' : 'Show Sources'}
          </button>
        </div>

        <div className={`${showSources ? 'grid grid-cols-2 gap-6' : ''} mb-12`}>
          <div className={`${showSources ? '' : 'w-full md:w-1/2 mx-auto'}`}>
            <div className="grid gap-6">
              {lessons.map((l, idx) => (
                <div
                  key={l.id}
                  className={`bg-white p-6 rounded-lg shadow ${
                    l.status==='done'    ? 'border-2 border-green-400' :
                    l.status==='pending' ? 'opacity-50' : ''
                  }`}
                >
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="text-xl font-semibold mb-1">
                        Lesson {idx + 1}: {l.title}
                      </h3>
                      <p className="text-gray-600">{l.summary}</p>
                    </div>
                    <div className="flex items-center gap-3">
                      {l.status==='done' && typeof l.score === 'number' && (
                        <div className={`px-3 py-1 rounded text-sm font-semibold ${
                          l.score >= 80 ? 'bg-green-100 text-green-800 border border-green-200' :
                          l.score >= 50 ? 'bg-yellow-100 text-yellow-800 border border-yellow-200' :
                                          'bg-red-100 text-red-800 border border-red-200'
                        }`}>
                          {l.score}%
                        </div>
                      )}
                      <button
                        onClick={() => startLesson(l)}
                        disabled={l.status==='pending'}
                        className={`px-6 py-2 rounded font-semibold ${
                          l.status==='done'  ? 'bg-green-600 text-white hover:bg-green-700' :
                          l.status==='ready' ? 'bg-blue-600 text-white hover:bg-blue-700' :
                                               'bg-gray-300 text-gray-500'
                        }`}
                      >
                        {l.status==='done'
                          ? 'Review Lesson'
                          : l.status==='ready'
                            ? 'Start Lesson'
                            : 'Creating…'
                        }
                      </button>
                    </div>
                  </div>
                  {l.status==='done' && l.score!=null && (
                    <div className="text-sm text-gray-700">
                      Score: {l.score}%
                    </div>
                  )}
                </div>
              ))}
              {lessons.length === 0 && (
                <div className="text-gray-600">No lessons found in this learning objective.</div>
              )}
            </div>
          </div>
          {showSources && (
            <div>
              <div className="bg-white p-4 rounded-lg shadow h-full">
                {sources.length === 0 ? (
                  <div className="text-gray-600">No source PDFs detected for this objective.</div>
                ) : (
                  <div className="grid gap-3">
                    {sources.map((s, i) => (
                      <div key={i} className="border rounded p-3">
                        <div className="flex items-center justify-between">
                          <div className="font-semibold">
                            {s.filename}{s.pdf_index!=null ? ` (PDF #${s.pdf_index+1})` : ''}
                          </div>
                          {s.pdf_id && (
                            <button
                              onClick={() => setActivePdfId(s.pdf_id)}
                              className="px-3 py-1 rounded border border-gray-300 bg-white hover:bg-gray-50"
                            >
                              View PDF
                            </button>
                          )}
                        </div>
                        <div className="text-sm text-gray-600 mt-1">
                          Chunks in LO: {s.chunk_indices.join(', ')}
                        </div>
                      </div>
                    ))}
                    {activePdfId && (
                      <div className="mt-2">
                        <div className="flex items-center justify-between mb-2">
                          <div className="font-semibold">Preview</div>
                          <button className="text-sm text-blue-600" onClick={() => setActivePdfId(null)}>Close</button>
                        </div>
                        <iframe
                          title="PDF Preview"
                          src={`${API_BASE}/api/pdf/${activePdfId}/raw`}
                          style={{ width: '100%', height: 520, border: '1px solid #e5e7eb', borderRadius: 6 }}
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
