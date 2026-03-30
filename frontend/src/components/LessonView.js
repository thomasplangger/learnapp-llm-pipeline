
import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import '../App.css';

const API_BASE = process.env.REACT_APP_BACKEND_URL || '';
const USER_ID  = 'default_user';

export default function LessonView() {
  const { courseId, lessonId } = useParams();
  const navigate               = useNavigate();

  
  const [lesson,         setLesson]     = useState(null);
  const [loading,        setLoading]    = useState(true);
  const [error,          setError]      = useState('');
  const [answers,        setAnswers]    = useState({});   
  const [evalResults,    setEvalResults]= useState({});   
  const [showEval,       setShowEval]   = useState({});   
  const [lessonDone,     setLessonDone] = useState(false);
  const [isReview,       setIsReview]   = useState(false);
  const [current,        setCurrent]    = useState(0);

  
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/lesson/${lessonId}`);
        if (!res.ok) throw new Error();
        const data = await res.json();
        if (!active) return;
        setLesson(data);
      } catch {
        if (active) setError('Failed to load lesson');
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => { active = false };
  }, [lessonId]);

  useEffect(() => {
    if (!lesson) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/api/progress/${USER_ID}`);
        if (!r.ok) return;
        const list = await r.json();
        const rec = Array.isArray(list) ? list.find(p => p.lesson_id === lessonId) : null;
        if (!rec || cancelled) return;
        const ans = rec.answers || {};
        setAnswers(ans || {});
        const shown = {};
        for (let i = 0; i < (lesson.questions?.length || 0); i++) shown[i] = true;
        setShowEval(shown);
        if (rec.completed) setIsReview(true);

        const evals = {};
        for (let i = 0; i < (lesson.questions?.length || 0); i++) {
          const q = lesson.questions[i];
          if (q?.type === 'TextQ' && ans[i]) {
            try {
              const er = await fetch(`${API_BASE}/api/evaluate-answer`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lesson_id: lessonId, question_index: i, user_answer: ans[i] })
              });
              if (er.ok) evals[i] = await er.json();
            } catch {}
          }
        }
        if (!cancelled && Object.keys(evals).length) setEvalResults(prev => ({ ...prev, ...evals }));
      } catch {}
    })();
    return () => { cancelled = true };
  }, [lesson, lessonId]);

  const saveProgress = useCallback(async (final = false) => {
    if (!lesson) return;
    let totalPts = 0, earned = 0;
    lesson.questions.forEach((q, i) => {
      const pts = q.points ?? (q.type === 'TextQ' ? 2 : 1);
      totalPts += pts;
      if (q.type === 'TextQ') {
        earned += evalResults[i]?.score || 0;
      } else {
        earned += answers[i] === q.correct_answer ? 1 : 0;
      }
    });

    const payload = {
      user_id:     USER_ID,
      lesson_id:   lessonId,
      course_id:   courseId,
      completed:   final,
      score:       final ? Math.round((earned/totalPts)*100) : null,
      time_spent:  0,
      completed_at: final ? new Date().toISOString() : null,
      answers
    };
    try {
      await fetch(`${API_BASE}/api/progress`, {
        method:  'POST',
        headers: { 'Content-Type':'application/json' },
        body:    JSON.stringify(payload)
      });
    } catch {
    }
  }, [answers, evalResults, lesson, lessonId, courseId]);

  useEffect(() => {
    if (lessonDone) saveProgress(true);
  }, [lessonDone, saveProgress]);

  const handleMCQ = choice =>
    setAnswers(a => ({ ...a, [current]: choice }));
  const handleText = e =>
    setAnswers(a => ({ ...a, [current]: e.target.value }));

  const doEvaluate = async () => {
    const q = lesson.questions[current];
    if (q.type === 'TextQ') {
      try {
        const res = await fetch(`${API_BASE}/api/evaluate-answer`, {
          method:  'POST',
          headers: { 'Content-Type':'application/json' },
          body:    JSON.stringify({
            lesson_id:      lessonId,
            question_index: current,
            user_answer:    answers[current] || ''
          })
        });
        const body = await res.json();
        setEvalResults(e => ({ ...e, [current]: body }));
        setShowEval(s => ({ ...s, [current]: true }));
        saveProgress(false);
      } catch {
        alert('Evaluation failed');
      }
    } else {
      setShowEval(s => ({ ...s, [current]: true }));
      saveProgress(false);
    }
  };

  const prevQ = () => {
    if (current > 0) setCurrent(c => c - 1);
  };
  const nextQ = () => {
    if (current < lesson.questions.length - 1) setCurrent(c => c + 1);
    else setLessonDone(true);
  };
  const backToCourse   = () => navigate(`/course/${courseId}`);
  const reviewQuestions = () => {
    setLessonDone(false);
    setCurrent(0);
    setIsReview(true);
  };

  if (loading) return <div>Loading…</div>;
  if (error)   return <div className="text-red-600">{error}</div>;
  if (!lesson) return null;

  const totalQ     = lesson.questions.length;
  const q          = lesson.questions[current];
  const ans        = answers[current] || '';
  const evaluated  = !!showEval[current];
  const readOnly   = evaluated || isReview;
  const ptsLabel   = q.type === 'TextQ' ? `${q.points} pts` : '1 pt';

  if (lessonDone) {
    let totalPts = 0, earned = 0;
    lesson.questions.forEach((qq,i) => {
      const p = qq.points ?? (qq.type === 'TextQ' ? 2 : 1);
      totalPts += p;
      if (qq.type === 'TextQ') earned += evalResults[i]?.score || 0;
      else earned += (answers[i] === qq.correct_answer ? 1 : 0);
    });

    return (
      <div className="min-h-screen bg-gray-50 p-8 text-center">
        <h2 className="text-3xl font-bold mb-4">🎉 Lesson Complete!</h2>
        <p className="text-xl mb-6">
          You scored {earned} / {totalPts} points.
        </p>
        <div className="flex justify-center space-x-4">
          <button
            onClick={backToCourse}
            className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
          >
            Back to Course
          </button>
          <button
            onClick={reviewQuestions}
            className="bg-gray-200 px-6 py-2 rounded hover:bg-gray-300"
          >
            Review Questions
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="flex justify-between mb-6">
        <button
          onClick={backToCourse}
          className="bg-gray-200 px-4 py-2 rounded hover:bg-gray-300"
        >
          Back
        </button>
        <div className="text-gray-600">
          Q {current + 1} / {totalQ} ({ptsLabel})
        </div>
      </div>

      <div className="bg-white p-6 rounded shadow mb-6">
        <h2 className="text-2xl font-bold mb-4">{lesson.title}</h2>
        <h3 className="text-xl mb-4">{q.question}</h3>

        {q.type !== 'TextQ' && Array.isArray(q.options) && (
          <div className="space-y-3 mb-6">
            {q.options.map((opt,i) => {
              const letter  = opt.charAt(0);
              const selected= answers[current] === letter;
              const correct = letter === q.correct_answer;
              let cls = 'border-gray-200 hover:border-gray-300';

              if (selected) {
                if (readOnly) {
                  cls = correct
                    ? 'border-green-500 bg-green-50'
                    : 'border-red-500 bg-red-50';
                } else {
                  cls = 'border-blue-500 bg-blue-50';
                }
              }

              return (
                <button
                  key={i}
                  onClick={() => !readOnly && handleMCQ(letter)}
                  disabled={readOnly}
                  className={`w-full text-left p-4 rounded border-2 ${cls}`}
                >
                  {opt}
                </button>
              );
            })}
          </div>
        )}

        {q.type === 'TextQ' && (
          <textarea
            rows={4}
            value={ans}
            onChange={handleText}
            disabled={readOnly}
            className="w-full border rounded p-2 mb-6"
          />
        )}

        {readOnly && (
          <div className={`p-4 mb-6 rounded border ${
            q.type === 'TextQ'
              ? ((evalResults[current]?.score || 0) === q.points
                  ? 'bg-green-50 border-green-200'
                  : evalResults[current]?.score > 0
                    ? 'bg-yellow-50 border-yellow-200'
                    : 'bg-red-50 border-red-200')
              : (answers[current] === q.correct_answer
                  ? 'bg-green-50 border-green-200'
                  : 'bg-red-50 border-red-200')
          }`}>
            <h4 className={`font-semibold mb-2 ${
              q.type === 'TextQ'
                ? ((evalResults[current]?.score || 0) === q.points
                    ? 'text-green-800'
                    : evalResults[current]?.score > 0
                      ? 'text-yellow-800'
                      : 'text-red-800')
                : (answers[current] === q.correct_answer
                    ? 'text-green-800'
                    : 'text-red-800')
            }`}>
              {q.type === 'TextQ'
                ? `Score: ${evalResults[current]?.score || 0}/${q.points}`
                : (answers[current] === q.correct_answer
                    ? '✅ Correct'
                    : '❌ Incorrect')}
            </h4>
            <p className="text-gray-700 mb-3">
              {q.type === 'TextQ'
                ? evalResults[current]?.explanation
                : q.explanation}
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <div className="text-sm text-gray-600">Your answer</div>
                <div className="p-3 bg-white rounded border">{(answers[current] || '').toString() || <em>—</em>}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600">Correct answer</div>
                <div className="p-3 bg-white rounded border">{(q.correct_answer || '').toString() || <em>—</em>}</div>
              </div>
            </div>
          </div>
        )}

        <div className="flex justify-between">
          <button
            onClick={prevQ}
            disabled={current === 0}
            className="px-4 py-2 rounded bg-gray-200 hover:bg-gray-300 disabled:opacity-50"
          >
            Previous
          </button>

          {!readOnly ? (
            <button
              onClick={doEvaluate}
              disabled={!ans}
              className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {q.type === 'TextQ' ? 'Evaluate' : 'Check Answer'}
            </button>
          ) : (
            <button
              onClick={nextQ}
              className="px-4 py-2 rounded bg-green-600 text-white hover:bg-green-700"
            >
              {current < totalQ - 1 ? 'Next Question' : 'Finish Lesson'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
