
"use client";

import { useState, type FormEvent } from "react";
import { Info, ClipboardList, RotateCcw } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PageShell } from "@/components/dashboard/page-shell";
import { clinicalFields } from "@/lib/clinical-schema";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "http://127.0.0.1:8000";
const ASSESSMENT_FEATURES = clinicalFields.map((field) => field.name);

interface AssessmentResponse {
  success: boolean;
  message: string;
  prediction: number;
  prediction_label: string;
  class_1_probability: number | null;
  model_connected: boolean;
  feature_count: number;
  processed_feature_count: number;
  model_name: string;
  disclaimer: string;
}

function isAssessmentResponse(
  value: unknown
): value is AssessmentResponse {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const response = value as Record<string, unknown>;
  return (
    typeof response.success === "boolean" &&
    typeof response.message === "string" &&
    typeof response.prediction === "number" &&
    typeof response.prediction_label === "string" &&
    (response.class_1_probability === null ||
      typeof response.class_1_probability === "number") &&
    typeof response.model_connected === "boolean" &&
    typeof response.feature_count === "number" &&
    typeof response.processed_feature_count === "number" &&
    typeof response.model_name === "string" &&
    typeof response.disclaimer === "string"
  );
}

function getErrorDetail(value: unknown): string | null {
  if (typeof value !== "object" || value === null || !("detail" in value)) {
    return null;
  }

  const detail = value.detail;
  if (typeof detail === "string") {
    return detail;
  }

  return JSON.stringify(detail);
}

export default function AssessmentPage() {
  const [values, setValues] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiResponse, setApiResponse] = useState<unknown>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (loading) {
      return;
    }

    setLoading(true);
    setError("");
    setApiResponse(null);

    try {
      if (
        ASSESSMENT_FEATURES.some(
          (feature) =>
            typeof values[feature] !== "string" ||
            values[feature].trim() === ""
        )
      ) {
        throw new Error("Complete every clinical field before submitting.");
      }

      const payload = Object.fromEntries(
        ASSESSMENT_FEATURES.map((feature) => [
          feature,
          Number(values[feature]),
        ])
      );

      if (
        Object.values(payload).some(
          (value) => typeof value !== "number" || !Number.isFinite(value)
        )
      ) {
        throw new Error("Complete every clinical field with a valid number.");
      }

      // Send patient data to FastAPI.
      const response = await fetch(
        `${API_BASE_URL.replace(/\/$/, "")}/assessment`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        }
      );

      let data: unknown;
      try {
        data = await response.json();
      } catch {
        throw new Error(
          `Backend returned HTTP ${response.status} without a valid JSON response.`
        );
      }

      if (!response.ok) {
        const detail = getErrorDetail(data);
        if (response.status === 404) {
          throw new Error(
            "HTTP 404: Assessment endpoint not found. Confirm the backend is running the current API at /assessment."
          );
        }
        if (response.status === 422) {
          throw new Error(
            `HTTP 422: Validation failed${detail ? ` — ${detail}` : "."}`
          );
        }
        if (response.status >= 500) {
          throw new Error(
            `HTTP ${response.status}: Backend failed to process the assessment${detail ? ` — ${detail}` : "."}`
          );
        }
        throw new Error(
          `HTTP ${response.status}: ${detail || "Backend request failed."}`
        );
      }

      if (!isAssessmentResponse(data)) {
        throw new Error(
          "Backend returned an invalid assessment response."
        );
      }

      setApiResponse(data);
    } catch (err) {
      setError(
        err instanceof TypeError
          ? `Network/CORS error: unable to connect to ${API_BASE_URL}. Confirm the backend is running and allows http://localhost:3000.`
          : err instanceof Error
            ? err.message
            : "Network/CORS error: unable to connect to the backend."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setValues({});
    setError("");
    setApiResponse(null);
  }

  return (
    <PageShell
      title="Patient Assessment"
      description="Enter clinical features for the research prototype."
    >
      <Badge className="bg-amber-100 text-amber-800 hover:bg-amber-100">
        ACADEMIC RESEARCH PROTOTYPE
      </Badge>

      <Card className="border-slate-200 shadow-sm">
        <CardContent className="p-6">
          <div className="mb-7 flex items-center gap-3">
            <div className="rounded-xl bg-blue-50 p-3">
              <ClipboardList className="text-blue-600" size={24} />
            </div>

            <div>
              <h2 className="font-semibold text-slate-900">
                Clinical Information
              </h2>
              <p className="text-sm text-slate-500">
                Initial Cleveland dataset feature schema
              </p>
            </div>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="grid gap-6 md:grid-cols-2">
              {clinicalFields.map((field) => (
                <div key={field.name} className="space-y-2">
                  <label
                    htmlFor={field.name}
                    className="text-sm font-medium text-slate-800"
                  >
                    {field.label}
                  </label>

                  <p className="text-xs text-slate-500">
                    {field.description}
                  </p>

                  {field.type === "select" ? (
                    <select
                      id={field.name}
                      name={field.name}
                      required
                      value={values[field.name] ?? ""}
                      onChange={(event) => {
                        setValues((current) => ({
                          ...current,
                          [field.name]: event.target.value,
                        }));
                        setApiResponse(null);
                        setError("");
                      }}
                      className="h-11 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                    >
                      <option value="">Select an option</option>

                      {field.options?.map((option) => (
                        <option
                          key={option.value}
                          value={option.value}
                        >
                          {option.label}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      id={field.name}
                      name={field.name}
                      type="number"
                      required
                      min={field.min}
                      max={field.max}
                      step={field.step ?? 1}
                      value={values[field.name] ?? ""}
                      onChange={(event) => {
                        setValues((current) => ({
                          ...current,
                          [field.name]: event.target.value,
                        }));
                        setApiResponse(null);
                        setError("");
                      }}
                      placeholder={`Enter ${field.label.toLowerCase()}`}
                      className="h-11 w-full rounded-lg border border-slate-200 px-3 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                    />
                  )}
                </div>
              ))}
            </div>

            <div className="mt-8 flex flex-wrap gap-3">
              <button
                type="submit"
                disabled={loading}
                className="rounded-lg bg-blue-600 px-5 py-3 text-sm font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? "Sending Data..." : "Submit Assessment"}
              </button>

              <button
                type="button"
                onClick={handleReset}
                disabled={loading}
                className="flex items-center gap-2 rounded-lg border border-slate-200 px-5 py-3 text-sm font-medium text-slate-600 hover:bg-slate-50"
              >
                <RotateCcw size={16} />
                Reset Form
              </button>
            </div>
          </form>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-red-200" role="alert">
          <CardContent className="p-5">
            <h3 className="font-semibold text-red-700">
              Backend Connection Error
            </h3>
            <p className="mt-2 text-sm text-red-600">
              {error}
            </p>
          </CardContent>
        </Card>
      )}

      {apiResponse !== null && (
        <Card className="border-green-200" role="status" aria-live="polite">
          <CardContent className="p-6">
            <h3 className="font-semibold text-green-700">
              Model Assessment Complete
            </h3>

            <p className="mt-1 text-sm text-slate-500">
              The saved model processed {isAssessmentResponse(apiResponse)
                ? apiResponse.feature_count
                : 0} clinical features.
            </p>

            {isAssessmentResponse(apiResponse) && (
              <div className="mt-4 space-y-2 rounded-lg bg-slate-50 p-4 text-sm text-slate-700">
                <p>{apiResponse.message}</p>
                <p>
                  Prediction:{" "}
                  <span className="font-medium">
                    {apiResponse.prediction_label}
                  </span>
                </p>
                <p>Model: <span className="font-medium">{apiResponse.model_name}</span></p>
                <p>
                  Class-1 model-estimated probability:{" "}
                  <span className="font-medium">
                    {apiResponse.class_1_probability === null
                      ? "Unavailable"
                      : `${(apiResponse.class_1_probability * 100).toFixed(1)}%`}
                  </span>
                </p>
                <p className="pt-2 text-xs text-slate-500">
                  {apiResponse.disclaimer}
                </p>
              </div>
            )}

            <pre className="mt-4 overflow-x-auto rounded-lg bg-slate-950 p-4 text-xs text-green-300">
              {JSON.stringify(apiResponse, null, 2)}
            </pre>

            <div className="mt-4 flex gap-2 rounded-lg bg-amber-50 p-4 text-sm text-amber-900">
              <Info className="shrink-0" size={19} />
              This result is a dataset classification label from an academic
              research prototype, not a medical diagnosis.
            </div>
          </CardContent>
        </Card>
      )}
    </PageShell>
  );
}
