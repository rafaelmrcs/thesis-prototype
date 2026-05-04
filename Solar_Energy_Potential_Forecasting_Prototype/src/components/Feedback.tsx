import { useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { MessageSquare, Send, CheckCircle2, Download, Upload, X } from 'lucide-react';

const FEEDBACK_URL = 'https://script.google.com/macros/s/AKfycbzei4tLFcm8Elzow4rlWNSgJx3UIeyST9lizZWQJM99F6FGlOrXGijN6Iy-Rb2DBTBTdg/exec';

const CRITERIA = [
  {
    group: 'Product Quality',
    items: [
      { key: 'pq_a', label: 'a. Reliability — proper functioning as specified and expected', weight: 5 },
      { key: 'pq_b', label: 'b. Robustness — acceptable response to unusual inputs, loads and conditions', weight: 5 },
      { key: 'pq_c', label: 'c. Efficiency of use by the frequent users', weight: 5 },
      { key: 'pq_d', label: 'd. Easy to use even for the frequent users', weight: 5 },
    ],
  },
  {
    group: 'Functional Testing',
    items: [
      { key: 'ft_a', label: 'a. The expected results occur when valid data is used', weight: 10 },
      { key: 'ft_b', label: 'b. The appropriate error / warning message are displayed when invalid data is used', weight: 10 },
    ],
  },
  {
    group: 'Functionality',
    items: [{ key: 'functionality', label: 'Suitability · Accuracy · Interoperability · Compliance · Security', weight: 10 }],
  },
  {
    group: 'Reliability',
    items: [{ key: 'reliability', label: 'Maturity · Fault-tolerance · Recoverability', weight: 10 }],
  },
  {
    group: 'Usability',
    items: [{ key: 'usability', label: 'Understandability · Learnability · Operability', weight: 10 }],
  },
  {
    group: 'Efficiency',
    items: [{ key: 'efficiency', label: 'Time Behaviour · Resource Behaviour', weight: 10 }],
  },
  {
    group: 'Maintainability',
    items: [{ key: 'maintainability', label: 'Effort needed to make specified modifications', weight: 10 }],
  },
  {
    group: 'Portability',
    items: [{ key: 'portability', label: 'Ability of software to be transferred from one environment to another', weight: 10 }],
  },
];

const ALL_KEYS = CRITERIA.flatMap((g) => g.items.map((i) => i.key));
const TOTAL_WEIGHT = CRITERIA.flatMap((g) => g.items).reduce((s, i) => s + i.weight, 0);

type ScoreMap = Record<string, string>;

export function Feedback() {
  const [info, setInfo] = useState({ name: '', email: '', role: '', comments: '' });
  const [scores, setScores] = useState<ScoreMap>(Object.fromEntries(ALL_KEYS.map((k) => [k, ''])));
  const [signaturePreview, setSignaturePreview] = useState<string | null>(null);
  const [signatureBase64, setSignatureBase64] = useState<string | null>(null);
  const [waiverAgreed, setWaiverAgreed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const setInfo$ = (field: keyof typeof info) =>
    (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
      setInfo((p) => ({ ...p, [field]: e.target.value }));

  const setScore = (key: string) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const weight = CRITERIA.flatMap((g) => g.items).find((i) => i.key === key)?.weight ?? 0;
    const raw = e.target.value;
    const num = parseFloat(raw);
    if (raw !== '' && (isNaN(num) || num < 0 || num > weight)) return;
    setScores((p) => ({ ...p, [key]: raw }));
  };

  const totalScore = ALL_KEYS.reduce((s, k) => s + (parseFloat(scores[k]) || 0), 0);

  const handleSignatureUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      setSignaturePreview(result);
      setSignatureBase64(result);
    };
    reader.readAsDataURL(file);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!info.name.trim()) {
      setError('Please enter your full name.');
      return;
    }
    if (!waiverAgreed) {
      setError('Please read and agree to the confidentiality notice before submitting.');
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      await fetch(FEEDBACK_URL, {
        method: 'POST',
        mode: 'no-cors',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: info.name.trim(),
          email: info.email.trim() || '—',
          role: info.role.trim() || '—',
          comments: info.comments.trim() || '—',
          scores,
          totalScore: totalScore.toFixed(2),
          signatureImage: signatureBase64 ?? '—',
          waiverAgreed: true,
        }),
      });
      setSubmitted(true);
    } catch {
      setError('Failed to submit. Please try again shortly.');
    } finally {
      setSubmitting(false);
    }
  };

  if (submitted) {
    return (
      <div className="max-w-2xl mx-auto">
        <Card className="border-2 border-emerald-200 bg-emerald-50 shadow-md">
          <CardContent className="pt-10 pb-10 flex flex-col items-center gap-4 text-center">
            <CheckCircle2 className="w-16 h-16 text-emerald-500" />
            <div>
              <p className="text-xl font-semibold text-emerald-800">Thank you for your evaluation!</p>
              <p className="text-sm text-emerald-700 mt-1">Your scores and feedback have been recorded. We appreciate your time.</p>
            </div>
            <Button
              variant="outline"
              className="mt-2 border-emerald-300 text-emerald-700 hover:bg-emerald-100"
              onClick={() => {
                setSubmitted(false);
                setScores(Object.fromEntries(ALL_KEYS.map((k) => [k, ''])));
                setInfo({ name: '', email: '', role: '', comments: '' });
                setSignaturePreview(null);
                setSignatureBase64(null);
                setWaiverAgreed(false);
              }}
            >
              Submit another response
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-3xl mx-auto">
      {/* Hero */}
      <div className="rounded-2xl bg-gradient-to-br from-violet-50 via-purple-50 to-fuchsia-50 border-2 border-violet-200 p-6 shadow-lg">
        <div className="flex items-start gap-4">
          <div className="w-14 h-14 bg-gradient-to-br from-violet-500 via-purple-500 to-fuchsia-500 rounded-xl flex items-center justify-center shadow-lg shrink-0">
            <MessageSquare className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Beta Testing Evaluation</h1>
            <p className="text-gray-600 mt-1 text-sm">
              Download the official form, score each criterion below, upload your signed form, and submit. Your evaluation helps us finalize this thesis prototype.
            </p>
          </div>
        </div>
      </div>

      {/* PDF Viewer */}
      <Card className="border-2 border-amber-200 shadow-md">
        <CardHeader>
          <div className="flex items-center justify-between gap-4">
            <div>
              <CardTitle className="text-lg">Official Beta Testing Form</CardTitle>
              <p className="text-xs text-slate-500 mt-0.5">F-16500-015 · University of Mindanao, College of Computing Education</p>
            </div>
            <a
              href="/beta-testing-form.pdf"
              download
              className="shrink-0 inline-flex items-center gap-2 rounded-lg bg-amber-500 hover:bg-amber-600 text-white text-sm font-medium px-4 py-2 shadow transition-colors"
            >
              <Download className="w-4 h-4" />
              Download
            </a>
          </div>
        </CardHeader>
        <CardContent className="p-0 pb-0">
          <iframe
            src="/beta-testing-form.pdf"
            className="w-full rounded-b-xl border-t border-amber-200"
            style={{ height: '600px' }}
            title="Beta Testing Criteria Form"
          />
        </CardContent>
      </Card>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Tester Info */}
        <Card className="border-2 border-violet-200 shadow-md">
          <CardHeader>
            <CardTitle className="text-lg">Tester Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="name" className="text-sm">Full Name *</Label>
              <Input id="name" placeholder="e.g., Juan dela Cruz" value={info.name} onChange={setInfo$('name')} />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <Label htmlFor="email" className="text-sm">Email Address</Label>
                <Input id="email" type="email" placeholder="e.g., juan@example.com" value={info.email} onChange={setInfo$('email')} />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="role" className="text-sm">Professional Role</Label>
                <Input id="role" placeholder="e.g., Engineer, Researcher, Student…" value={info.role} onChange={setInfo$('role')} />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Scoring Table */}
        <Card className="border-2 border-slate-200 shadow-md">
          <CardHeader>
            <CardTitle className="text-lg">Beta Testing Scores</CardTitle>
            <p className="text-xs text-slate-500 mt-1">Enter your score for each criterion. Max score per item equals its weight.</p>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto rounded-xl border border-slate-200">
              <table className="w-full text-sm">
                <thead className="bg-slate-100 text-slate-700">
                  <tr>
                    <th className="px-4 py-3 text-left font-semibold">Criterion</th>
                    <th className="px-4 py-3 text-center font-semibold w-24">Weight</th>
                    <th className="px-4 py-3 text-center font-semibold w-28">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {CRITERIA.map((group) => (
                    <>
                      <tr key={group.group} className="bg-violet-50">
                        <td colSpan={3} className="px-4 py-2 font-semibold text-violet-800 text-xs uppercase tracking-wide">
                          {group.group}
                        </td>
                      </tr>
                      {group.items.map((item) => (
                        <tr key={item.key} className="border-t border-slate-100 hover:bg-slate-50">
                          <td className="px-4 py-3 text-slate-700">{item.label}</td>
                          <td className="px-4 py-3 text-center text-slate-500">{item.weight}%</td>
                          <td className="px-4 py-3 text-center">
                            <input
                              type="number"
                              min={0}
                              max={item.weight}
                              step={0.5}
                              value={scores[item.key]}
                              onChange={setScore(item.key)}
                              placeholder="—"
                              className="w-20 rounded-md border border-slate-300 bg-white px-2 py-1 text-center text-sm focus:outline-none focus:ring-1 focus:ring-violet-400"
                            />
                          </td>
                        </tr>
                      ))}
                    </>
                  ))}
                  <tr className="border-t-2 border-slate-300 bg-slate-50 font-semibold">
                    <td className="px-4 py-3">Total</td>
                    <td className="px-4 py-3 text-center">{TOTAL_WEIGHT}%</td>
                    <td className="px-4 py-3 text-center text-violet-700 text-base">{totalScore.toFixed(2)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Comments */}
        <Card className="border-2 border-slate-200 shadow-md">
          <CardHeader>
            <CardTitle className="text-lg">Comments / Recommendations</CardTitle>
          </CardHeader>
          <CardContent>
            <textarea
              rows={4}
              placeholder="Share any additional observations or recommendations about the system…"
              value={info.comments}
              onChange={setInfo$('comments')}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring resize-none"
            />
          </CardContent>
        </Card>

        {/* Signature Upload */}
        <Card className="border-2 border-slate-200 shadow-md">
          <CardHeader>
            <CardTitle className="text-lg">Signed Form / Signature Image</CardTitle>
            <p className="text-xs text-slate-500 mt-1">Upload a photo or scan of your signed beta testing form, or just your signature.</p>
          </CardHeader>
          <CardContent>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleSignatureUpload}
            />
            {signaturePreview ? (
              <div className="relative inline-block">
                <img src={signaturePreview} alt="Signature preview" className="max-h-48 rounded-xl border border-slate-200 shadow-sm object-contain" />
                <button
                  type="button"
                  onClick={() => { setSignaturePreview(null); setSignatureBase64(null); if (fileInputRef.current) fileInputRef.current.value = ''; }}
                  className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-red-500 text-white flex items-center justify-center shadow hover:bg-red-600"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="flex flex-col items-center justify-center w-full h-32 rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 hover:bg-slate-100 hover:border-violet-400 transition-colors gap-2 text-slate-500 hover:text-violet-600"
              >
                <Upload className="w-6 h-6" />
                <span className="text-sm font-medium">Click to upload image</span>
                <span className="text-xs">PNG, JPG, JPEG accepted</span>
              </button>
            )}
          </CardContent>
        </Card>

        {/* Confidentiality Waiver */}
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-5 space-y-3 text-xs text-slate-600 leading-relaxed">
          <p className="font-semibold text-slate-800 text-sm">Confidentiality &amp; Data Privacy Notice</p>
          <p>
            By submitting this form, you confirm that the scores and feedback provided reflect your honest and independent evaluation of the system. Your personal information (name, email address) will be used solely to verify your participation in the beta testing of this academic thesis prototype — <strong>An Optimized AdaBoost Regression Algorithm with Feature-Aware Weighting for Solar Energy Potential Forecasting</strong> — and will not be shared with any third party.
          </p>
          <p>
            Submitted responses, including scores and comments, may be included in aggregate or anonymized form in the thesis research report and related academic outputs. This prototype and its associated research are the intellectual property of Mercado, Retardo, and Verzosa — University of Mindanao, College of Computing Education, Academic Year 2026.
          </p>
          <p>
            Uploaded signature images are collected solely as proof of evaluation participation and will be stored securely. They will not be used for any purpose beyond academic documentation of this beta testing activity.
          </p>
          <label className="flex items-start gap-3 mt-2 cursor-pointer">
            <input
              type="checkbox"
              checked={waiverAgreed}
              onChange={(e) => setWaiverAgreed(e.target.checked)}
              className="mt-0.5 h-4 w-4 rounded border-slate-300 accent-violet-600 cursor-pointer"
            />
            <span className="text-slate-700 font-medium">I have read and agree to the confidentiality and data privacy notice above.</span>
          </label>
        </div>

        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        <Button
          type="submit"
          disabled={submitting}
          className="w-full bg-gradient-to-r from-violet-500 to-fuchsia-500 hover:from-violet-600 hover:to-fuchsia-600 text-white shadow-md"
          size="lg"
        >
          {submitting ? (
            <>
              <div className="w-4 h-4 mr-2 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Submitting…
            </>
          ) : (
            <>
              <Send className="w-4 h-4 mr-2" />
              Submit Evaluation
            </>
          )}
        </Button>
      </form>
    </div>
  );
}
