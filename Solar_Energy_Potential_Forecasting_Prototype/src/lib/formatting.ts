export function formatExact(value: number, fractionDigits = 6): string {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
    useGrouping: false,
  });
}

const KWH_TO_J = 3_600_000;

export function formatResultsErrorMetric(valueInKwh: number): string {
  return formatExact(valueInKwh * KWH_TO_J, 6);
}

export function formatRawR2(value: number): string {
  return formatExact(value, 6);
}

export function formatPercent(value: number, fractionDigits = 6): string {
  return `${formatExact(value, fractionDigits)}%`;
}
