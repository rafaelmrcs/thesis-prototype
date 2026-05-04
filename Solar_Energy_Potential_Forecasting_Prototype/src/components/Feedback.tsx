import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { MessageSquare, Send, CheckCircle2 } from 'lucide-react';

const FEEDBACK_URL = 'https://script.google.com/macros/s/AKfycbzei4tLFcm8Elzow4rlWNSgJx3UIeyST9lizZWQJM99F6FGlOrXGijN6Iy-Rb2DBTBTdg/exec';

export function Feedback() {
  const [form, setForm] = useState({ name: '', email: '', role: '', comments: '' });
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = (field: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) =>
    setForm((prev) => ({ ...prev, [field]: e.target.value }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.comments.trim()) {
      setError('Please enter your comments or recommendations before submitting.');
      return;
    }
    if (!FEEDBACK_URL) {
      setError('Feedback endpoint is not configured. Set VITE_FEEDBACK_URL in your environment.');
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
          name: form.name.trim() || 'Anonymous',
          email: form.email.trim() || '—',
          role: form.role || '—',
          comments: form.comments.trim(),
        }),
      });
      setSubmitted(true);
      setForm({ name: '', email: '', role: '', comments: '' });
    } catch {
      setError('Failed to submit. Please try again shortly.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-8 max-w-2xl mx-auto">
      {/* Hero */}
      <div className="rounded-2xl bg-gradient-to-br from-violet-50 via-purple-50 to-fuchsia-50 border-2 border-violet-200 p-6 shadow-lg">
        <div className="flex items-start gap-4">
          <div className="w-14 h-14 bg-gradient-to-br from-violet-500 via-purple-500 to-fuchsia-500 rounded-xl flex items-center justify-center shadow-lg shrink-0">
            <MessageSquare className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Share Your Thoughts</h1>
            <p className="text-gray-600 mt-1 text-sm">
              Your feedback helps us improve this prototype. Share your observations, recommendations, or any comments about the solar energy forecasting system.
            </p>
          </div>
        </div>
      </div>

      {submitted ? (
        <Card className="border-2 border-emerald-200 bg-emerald-50 shadow-md">
          <CardContent className="pt-8 pb-8 flex flex-col items-center gap-4 text-center">
            <CheckCircle2 className="w-14 h-14 text-emerald-500" />
            <div>
              <p className="text-xl font-semibold text-emerald-800">Thank you for your feedback!</p>
              <p className="text-sm text-emerald-700 mt-1">Your response has been recorded. We appreciate your time.</p>
            </div>
            <Button
              variant="outline"
              className="mt-2 border-emerald-300 text-emerald-700 hover:bg-emerald-100"
              onClick={() => setSubmitted(false)}
            >
              Submit another response
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Card className="border-2 border-violet-200 shadow-md">
          <CardHeader>
            <CardTitle className="text-xl">Feedback Form</CardTitle>
            <CardDescription>Fields marked with * are required.</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-5">
              {/* Name */}
              <div className="space-y-1.5">
                <Label htmlFor="name" className="text-sm">Full Name</Label>
                <Input
                  id="name"
                  placeholder="e.g., Juan dela Cruz"
                  value={form.name}
                  onChange={set('name')}
                />
              </div>

              {/* Email */}
              <div className="space-y-1.5">
                <Label htmlFor="email" className="text-sm">Email Address</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="e.g., juan@example.com"
                  value={form.email}
                  onChange={set('email')}
                />
              </div>

              {/* Professional Role */}
              <div className="space-y-1.5">
                <Label htmlFor="role" className="text-sm">Professional Role</Label>
                <Input
                  id="role"
                  placeholder="e.g., Engineer, Researcher, Student…"
                  value={form.role}
                  onChange={set('role')}
                />
              </div>

              {/* Comments */}
              <div className="space-y-1.5">
                <Label htmlFor="comments" className="text-sm">Comments / Recommendations *</Label>
                <textarea
                  id="comments"
                  rows={5}
                  placeholder="Share your observations, suggestions, or recommendations about the system…"
                  value={form.comments}
                  onChange={set('comments')}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring resize-none"
                />
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
                    Submit Feedback
                  </>
                )}
              </Button>
            </form>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
