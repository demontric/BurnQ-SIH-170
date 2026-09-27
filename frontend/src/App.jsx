import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'
import ReactECharts from 'echarts-for-react'
import {
  Activity,
  AlertTriangle,
  ChevronDown,
  Crosshair,
  Info,
  LineChart,
  Loader2,
  Microscope,
  TableProperties,
  Target,
  Upload,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

const API_URL =
  import.meta.env.VITE_API_URL ?? '/api/detect-anomaly'
const TIME_LABELS = ['0h', '24h', '96h', '168h']

function normalizeStatus(value) {
  const status = String(value ?? '').trim().toUpperCase()
  return ['PASS', 'WATCH', 'REJECT'].includes(status) ? status : null
}

function isFlagged(row) {
  const status = normalizeStatus(row?.status)
  if (status === 'PASS') return false
  if (status === 'WATCH' || status === 'REJECT') return true
  return Boolean(row?.is_anomaly || row?.safety_slope_exceeded)
}

function lotKey(value) {
  if (value == null || value === '') return '—'
  return String(value).trim()
}

const PARAMETER_META = {
  leakage_current: { label: 'Leakage Current', unit: 'µA' },
  leakage: { label: 'Leakage Current', unit: 'µA' },
  iddq: { label: 'IDDQ', unit: 'µA' },
  prop_delay: { label: 'Propagation Delay', unit: 'ns' },
  propagation_delay: { label: 'Propagation Delay', unit: 'ns' },
}

const REASON_MESSAGES = {
  ABSOLUTE_LIMIT_EXCEEDED: 'Measured value exceeds the datasheet limit.',
  LOT_RELATIVE_OUTLIER: 'Parameter is significantly above the normal distribution for its lot.',
  PREDICTED_LIMIT_BREACH: 'Predicted 168h value exceeds the datasheet limit.',
  SAFETY_SLOPE_EXCEEDED: 'Predicted drift exceeds the calibrated safety slope.',
}

function humanizeParameter(value) {
  if (!value) return 'Unknown'
  const key = String(value).toLowerCase()
  if (PARAMETER_META[key]) return PARAMETER_META[key].label
  return String(value)
    .split(/[_\s]+/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

function parameterUnit(parameter) {
  const key = String(parameter ?? '').toLowerCase()
  return PARAMETER_META[key]?.unit ?? 'units'
}

function getCommonUnit(rows) {
  const units = [...new Set(rows.map((row) => parameterUnit(row.parameter)))]
  return units.length === 1 ? units[0] : 'native units'
}

function toFiniteNumber(value, fallback = null) {
  if (value == null || value === '') return fallback
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : fallback
}

function normalizeReasonCodes(value) {
  if (Array.isArray(value)) {
    return value
      .map((code) => String(code ?? '').trim())
      .filter(Boolean)
  }

  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return []

    if (trimmed.startsWith('[')) {
      try {
        const parsed = JSON.parse(trimmed)
        if (Array.isArray(parsed)) return normalizeReasonCodes(parsed)
      } catch {
        // Fall through and treat the value as one reason code.
      }
    }

    return [trimmed]
  }

  return []
}

function reasonText(reasonCodes) {
  const codes = normalizeReasonCodes(reasonCodes)
  if (codes.length === 0) return ''
  return codes
    .map((code) => REASON_MESSAGES[code] ?? `Flagged for rule: ${code}.`)
    .join(' ')
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;')
}

function formatValue(value, parameter, decimals = 2) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return `${Number(value).toFixed(decimals)} ${parameterUnit(parameter)}`
}

function formatLimit(value, parameter, decimals = 2) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return `${Number(value).toFixed(decimals)} ${parameterUnit(parameter)}`
}

function formatSignedValue(value, parameter, decimals = 2) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  const numeric = Number(value)
  return `${numeric >= 0 ? '+' : ''}${numeric.toFixed(decimals)} ${parameterUnit(parameter)}`
}

function normalizeRow(row) {
  const parameter = row.parameter ?? row.Parameter ?? 'unknown'
  const statusFromApi = normalizeStatus(row.status)
  const reason_codes = normalizeReasonCodes(row.reason_codes)
  const status = statusFromApi ?? (row.is_anomaly || row.safety_slope_exceeded ? 'WATCH' : 'PASS')
  const backendJustification =
    typeof row.justification === 'string' && row.justification.trim()
      ? row.justification.trim()
      : ''

  return {
    ComponentID:
      row.ComponentID ?? row.component_id ?? row.part_id ?? row.PartID ?? '—',
    Lot: lotKey(row.Lot ?? row.lot_id ?? row.lot),
    parameter,
    Value_0h: toFiniteNumber(row.Value_0h ?? row.value_0h, 0),
    Value_24h: toFiniteNumber(row.Value_24h ?? row.value_24h, 0),
    Value_96h: toFiniteNumber(row.Value_96h ?? row.value_96h, 0),
    Value_168h: toFiniteNumber(row.Value_168h ?? row.value_168h, 0),
    predicted_168h:
      row.predicted_168h != null ? toFiniteNumber(row.predicted_168h) : null,
    robust_z_score:
      row.robust_z_score != null ? toFiniteNumber(row.robust_z_score) : null,
    isolation_forest_score:
      row.isolation_forest_score != null
        ? toFiniteNumber(row.isolation_forest_score)
        : null,
    predicted_drift_rate:
      row.predicted_drift_rate != null
        ? toFiniteNumber(row.predicted_drift_rate)
        : null,
    safety_slope:
      row.safety_slope != null ? toFiniteNumber(row.safety_slope) : null,
    datasheet_limit:
      row.datasheet_limit != null ? toFiniteNumber(row.datasheet_limit) : null,
    is_anomaly: Boolean(row.is_anomaly),
    safety_slope_exceeded: Boolean(row.safety_slope_exceeded),
    status,
    reason_codes,
    justification:
      backendJustification ||
      reasonText(reason_codes) ||
      (status === 'PASS'
        ? 'Component passed screening.'
        : 'Component was flagged by the screening pipeline.'),
  }
}

function parseApiResponse(payload) {
  const rows = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.data)
      ? payload.data
      : []

  const data = rows.map(normalizeRow)
  const flagged_count =
    payload?.flagged_count ?? data.filter(isFlagged).length

  return {
    components: payload?.components ?? payload?.total_components ?? data.length,
    parametric_records:
      payload?.parametric_records ?? payload?.total_components ?? data.length,
    flagged_count,
    mae: payload?.mae != null ? toFiniteNumber(payload.mae) : null,
    mae_by_parameter:
      payload?.mae_by_parameter && typeof payload.mae_by_parameter === 'object'
        ? payload.mae_by_parameter
        : {},
    data,
  }
}

function computeMae(rows) {
  const pairs = rows.filter(
    (row) =>
      row.predicted_168h != null &&
      !Number.isNaN(row.predicted_168h) &&
      !Number.isNaN(row.Value_168h),
  )
  if (pairs.length === 0) return null
  const total = pairs.reduce(
    (sum, row) => sum + Math.abs(row.predicted_168h - row.Value_168h),
    0,
  )
  return total / pairs.length
}

function formatZScore(value) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return Number(value).toFixed(2)
}

function getStatusBadge(row) {
  const status = normalizeStatus(row.status) ?? 'PASS'

  if (status === 'REJECT') {
    return (
      <Badge variant="destructive" className="bg-red-500/20 text-red-300">
        REJECT
      </Badge>
    )
  }

  if (status === 'WATCH') {
    return (
      <Badge
        variant="outline"
        className="border-amber-500/50 bg-amber-500/15 text-amber-300"
      >
        WATCH
      </Badge>
    )
  }

  return (
    <Badge variant="secondary" className="bg-slate-800 text-slate-300">
      PASS
    </Badge>
  )
}

function getFlaggedColor(row) {
  const status = normalizeStatus(row.status)
  if (status === 'REJECT') return '#ef4444'
  if (status === 'WATCH') return '#f59e0b'
  return '#94a3b8'
}

function buildChartOption(rows, mode = 'raw', activeLimit = null) {
  const series = []
  const legendEntries = []
  const tooltipMeta = []
  const isPercent = mode === 'percent'

  const scaleFor = (row) => {
    if (!isPercent) return 1
    const limit = row.datasheet_limit ?? activeLimit
    return limit > 0 ? 100 / limit : 1
  }

  const yAxisUnit = getCommonUnit(rows)

  rows.forEach((row) => {
    const flagged = isFlagged(row)
    const scale = scaleFor(row)
    const trajectory = [
      row.Value_0h * scale,
      row.Value_24h * scale,
      row.Value_96h * scale,
      row.Value_168h * scale,
    ]

    series.push({
      name: row.ComponentID,
      type: 'line',
      data: trajectory,
      symbol: flagged ? 'circle' : 'none',
      symbolSize: flagged ? 6 : 0,
      lineStyle: {
        color: flagged ? getFlaggedColor(row) : '#64748b',
        width: flagged ? 2.5 : 1,
        opacity: flagged ? 0.85 : 0.15,
      },
      itemStyle: {
        color: getFlaggedColor(row),
        opacity: flagged ? 0.85 : 0.15,
      },
      emphasis: {
        lineStyle: { width: flagged ? 3.5 : 1.5, opacity: 1 },
      },
      z: flagged ? 10 : 1,
    })
    tooltipMeta.push({ row, predicted: false })

    if (flagged && row.predicted_168h != null) {
      series.push({
        name: `${row.ComponentID} (predicted)`,
        type: 'line',
        data: [null, null, row.Value_96h * scale, row.predicted_168h * scale],
        symbol: ['none', 'none', 'circle', 'diamond'],
        symbolSize: 7,
        lineStyle: {
          color: getFlaggedColor(row),
          width: 2,
          type: 'dashed',
          opacity: 0.85,
        },
        itemStyle: { color: getFlaggedColor(row) },
        z: 11,
      })
      tooltipMeta.push({ row, predicted: true })
      legendEntries.push(`${row.ComponentID} (predicted)`)
    }

    if (flagged) legendEntries.push(row.ComponentID)
  })

  const limitLine = isPercent
    ? { value: 100, label: 'Datasheet Limit (100%)' }
    : activeLimit != null
      ? {
          value: activeLimit,
          label: `Datasheet Limit (${formatLimit(activeLimit, rows[0]?.parameter)})`,
        }
      : null

  return {
    backgroundColor: 'transparent',
    animation: rows.length < 300,
    grid: { left: 56, right: 24, top: 48, bottom: 48 },
    tooltip: {
      trigger: 'item',
      backgroundColor: '#0f172a',
      borderColor: '#334155',
      textStyle: { color: '#e2e8f0', fontSize: 12 },
      formatter(params) {
        const meta = tooltipMeta[params?.seriesIndex]
        if (!meta?.row) return ''

        const row = meta.row
        const label = TIME_LABELS[params.dataIndex] ?? ''
        const value =
          params.value != null && !Number.isNaN(Number(params.value))
            ? isPercent
              ? `${Number(params.value).toFixed(1)}% of limit`
              : formatValue(Number(params.value), row.parameter)
            : '—'
        const status = escapeHtml(row.status)
        const parameter = escapeHtml(humanizeParameter(row.parameter))
        const justification = escapeHtml(row.justification)
        const reasonCodes = row.reason_codes.length
          ? `<br/><span style="color:#fbbf24">Reason:</span> ${escapeHtml(reasonText(row.reason_codes))}`
          : ''

        return (
          `<strong>${escapeHtml(row.ComponentID)}</strong><br/>` +
          `Parameter: ${parameter}<br/>` +
          `Status: <strong>${status}</strong><br/>` +
          `${label}: ${escapeHtml(value)}<br/>` +
          `Prediction: ${meta.predicted ? 'Module B 168h forecast' : 'Measured trajectory'}<br/>` +
          `<span style="color:#cbd5e1">${justification}</span>` +
          reasonCodes
        )
      },
    },
    legend: {
      type: 'scroll',
      top: 8,
      right: 16,
      textStyle: { color: '#94a3b8', fontSize: 11 },
      data: [...new Set(legendEntries)],
      show: legendEntries.length > 0 && legendEntries.length <= 20,
    },
    xAxis: {
      type: 'category',
      data: TIME_LABELS,
      axisLine: { lineStyle: { color: '#475569' } },
      axisLabel: { color: '#94a3b8' },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      name: isPercent
        ? '% of Datasheet Limit'
        : `Parameter Value (${yAxisUnit})`,
      nameTextStyle: { color: '#94a3b8', padding: [0, 0, 0, 8] },
      axisLine: { show: false },
      axisLabel: {
        color: '#94a3b8',
        formatter: isPercent ? '{value}%' : '{value}',
      },
      splitLine: { lineStyle: { color: '#1e293b', type: 'dashed' } },
    },
    series: [
      ...series,
      ...(limitLine
        ? [
            {
              name: 'Datasheet Limit',
              type: 'line',
              data: [],
              markLine: {
                silent: true,
                symbol: 'none',
                lineStyle: { color: '#fb923c', type: 'dotted', width: 2 },
                label: {
                  formatter: limitLine.label,
                  color: '#fdba74',
                  position: 'insideEndTop',
                },
                data: [{ yAxis: limitLine.value }],
              },
            },
          ]
        : []),
    ],
  }
}

function niceAxisBounds(minValue, maxValue) {
  const rawMin = Math.max(0, Number(minValue) || 0)
  const rawMax = Math.max(rawMin, Number(maxValue) || rawMin + 1)
  const paddedMin = Math.max(0, rawMin - (rawMax - rawMin) * 0.05)
  const paddedMax = rawMax + (rawMax - rawMin) * 0.05
  const range = Math.max(paddedMax - paddedMin, 1e-9)
  const roughStep = range / 6
  const magnitude = 10 ** Math.floor(Math.log10(roughStep))
  const normalized = roughStep / magnitude
  const stepMultiplier = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10
  const step = magnitude * stepMultiplier
  const lo = Math.max(0, Math.floor(paddedMin / step) * step)
  const hi = Math.ceil(paddedMax / step) * step
  return { lo, hi: hi > lo ? hi : lo + step, step }
}

function buildParityChartOption(rows) {
  const points = rows.filter(
    (row) =>
      row.predicted_168h != null &&
      !Number.isNaN(Number(row.predicted_168h)) &&
      !Number.isNaN(Number(row.Value_168h)),
  )

  if (points.length === 0) {
    return {
      backgroundColor: 'transparent',
      title: {
        text: 'No prediction data available',
        left: 'center',
        top: 'middle',
        textStyle: { color: '#64748b', fontSize: 13, fontWeight: 'normal' },
      },
    }
  }

  const allValues = points.flatMap((row) => [row.Value_168h, row.predicted_168h])
  const minValue = Math.min(...allValues)
  const maxValue = Math.max(...allValues)
  const { lo, hi, step } = niceAxisBounds(minValue, maxValue)
  const commonUnit = getCommonUnit(points)

  const flaggedPoints = []
  const passPoints = []

  points.forEach((row) => {
    const point = {
      value: [row.Value_168h, row.predicted_168h],
      name: row.ComponentID,
      error: row.predicted_168h - row.Value_168h,
      row,
    }

    if (isFlagged(row)) {
      flaggedPoints.push({ ...point, itemStyle: { color: getFlaggedColor(row) } })
    } else {
      passPoints.push(point)
    }
  })

  return {
    backgroundColor: 'transparent',
    animation: points.length < 500,
    grid: { left: 64, right: 24, top: 24, bottom: 56 },
    tooltip: {
      trigger: 'item',
      backgroundColor: '#0f172a',
      borderColor: '#334155',
      textStyle: { color: '#e2e8f0', fontSize: 12 },
      formatter(params) {
        if (!params?.data?.row || params.seriesType !== 'scatter') return ''
        const row = params.data.row
        const [actual, predicted] = params.data.value
        const error = params.data.error
        const reasonCodes = row.reason_codes.length
          ? `<br/><span style="color:#fbbf24">Reason:</span> ${escapeHtml(reasonText(row.reason_codes))}`
          : ''

        return (
          `<strong>${escapeHtml(row.ComponentID)}</strong><br/>` +
          `Parameter: ${escapeHtml(humanizeParameter(row.parameter))}<br/>` +
          `Status: <strong>${escapeHtml(row.status)}</strong><br/>` +
          `Actual 168h: ${escapeHtml(formatValue(actual, row.parameter))}<br/>` +
          `Predicted 168h: ${escapeHtml(formatValue(predicted, row.parameter))}<br/>` +
          `Error: ${escapeHtml(formatSignedValue(error, row.parameter))}<br/>` +
          `${escapeHtml(row.justification)}` +
          reasonCodes
        )
      },
    },
    xAxis: {
      type: 'value',
      name: `Actual 168h (${commonUnit})`,
      nameLocation: 'middle',
      nameGap: 32,
      nameTextStyle: { color: '#94a3b8' },
      min: lo,
      max: hi,
      interval: step,
      axisLine: { lineStyle: { color: '#475569' } },
      axisLabel: {
        color: '#94a3b8',
        formatter: (value) => Number(value).toFixed(step < 1 ? 1 : 0),
      },
      splitLine: { lineStyle: { color: '#1e293b', type: 'dashed' } },
    },
    yAxis: {
      type: 'value',
      name: `Predicted 168h (${commonUnit})`,
      nameTextStyle: { color: '#94a3b8', padding: [0, 0, 0, 8] },
      min: lo,
      max: hi,
      interval: step,
      axisLine: { show: false },
      axisLabel: {
        color: '#94a3b8',
        formatter: (value) => Number(value).toFixed(step < 1 ? 1 : 0),
      },
      splitLine: { lineStyle: { color: '#1e293b', type: 'dashed' } },
    },
    series: [
      {
        name: 'Perfect prediction',
        type: 'line',
        data: [
          [lo, lo],
          [hi, hi],
        ],
        showSymbol: false,
        lineStyle: { color: '#fb923c', type: 'dashed', width: 1.5, opacity: 0.7 },
        tooltip: { show: false },
        z: 1,
      },
      {
        name: 'Pass',
        type: 'scatter',
        data: passPoints,
        symbolSize: 6,
        itemStyle: { color: '#64748b', opacity: 0.35 },
        z: 2,
      },
      {
        name: 'Flagged',
        type: 'scatter',
        data: flaggedPoints,
        symbolSize: 8,
        itemStyle: { opacity: 0.9 },
        z: 3,
      },
    ],
  }
}

function SummaryCard({ title, value, description, icon: Icon, accent }) {
  return (
    <Card className="border-slate-800 bg-slate-900/70 ring-slate-800">
      <CardHeader className="gap-1 pb-0">
        <div className="flex items-center justify-between">
          <CardDescription className="text-slate-400">{title}</CardDescription>
          <div
            className={`p-1.5 ${accent ?? 'bg-slate-800 text-slate-300'}`}
          >
            <Icon className="size-3.5" />
          </div>
        </div>
        <CardTitle className="text-2xl font-semibold tracking-tight text-slate-50">
          {value}
        </CardTitle>
        <p className="text-xs text-slate-500">{description}</p>
      </CardHeader>
    </Card>
  )
}

function TabButton({ active, onClick, icon: Icon, label, count }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex h-9 items-center gap-2 px-3 text-sm font-medium transition-colors ${
        active
          ? 'bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/30'
          : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
      }`}
    >
      <Icon className="size-4" />
      {label}
      {count != null && (
        <span className="bg-slate-800 px-1.5 py-0.5 text-xs text-slate-300">
          {count} records
        </span>
      )}
    </button>
  )
}

// Loading indicator
function BouncingDots() {
  return (
    <div className="flex items-center justify-center space-x-2">
      <div className="size-2.5 rounded-full bg-amber-500 animate-bounce [animation-delay:-0.3s]"></div>
      <div className="size-2.5 rounded-full bg-amber-500 animate-bounce [animation-delay:-0.15s]"></div>
      <div className="size-2.5 rounded-full bg-amber-500 animate-bounce"></div>
    </div>
  )
}

function ComponentRegistry({
  rows,
  expandedRowId,
  onToggleRow,
  loading,
  hasResults,
}) {
  if (loading) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 py-20">
        <BouncingDots />
        <p className="text-sm font-medium text-amber-500/80 animate-pulse">
          Running anomaly models and screening rules...
        </p>
      </div>
    )
  }

  if (!hasResults || rows.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-slate-500">
        No component data loaded.
      </p>
    )
  }

  return (
    <Table>
      <TableHeader className="sticky top-0 z-10 bg-slate-900">
        <TableRow className="border-slate-800 hover:bg-transparent">
          <TableHead className="text-slate-400">ID</TableHead>
          <TableHead className="text-slate-400">Lot</TableHead>
          <TableHead className="text-slate-400">0h</TableHead>
          <TableHead className="text-slate-400">24h</TableHead>
          <TableHead className="text-slate-400">96h</TableHead>
          <TableHead className="text-slate-400">168h</TableHead>
          <TableHead className="text-slate-400">Pred 168h</TableHead>
          <TableHead className="text-slate-400">Z-Score</TableHead>
          <TableHead className="text-slate-400">Status</TableHead>
          <TableHead className="w-10 text-slate-400" />
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((row, index) => {
          const rowKey = `${row.ComponentID}-${row.Lot}-${index}`
          const isExpanded = expandedRowId === rowKey
          const flagged = isFlagged(row)

          return (
            <Fragment key={rowKey}>
              <TableRow
                onClick={() => onToggleRow(rowKey, row)}
                className={`cursor-pointer border-slate-800 ${
                  flagged
                    ? 'bg-red-500/5 hover:bg-red-500/10'
                    : 'hover:bg-slate-800/50'
                } ${isExpanded ? 'bg-slate-800/60' : ''}`}
              >
                <TableCell className="font-medium text-slate-200">
                  {row.ComponentID}
                </TableCell>
                <TableCell className="text-slate-300">{row.Lot}</TableCell>
                <TableCell className="text-slate-300">
                  {formatValue(row.Value_0h, row.parameter)}
                </TableCell>
                <TableCell className="text-slate-300">
                  {formatValue(row.Value_24h, row.parameter)}
                </TableCell>
                <TableCell className="text-slate-300">
                  {formatValue(row.Value_96h, row.parameter)}
                </TableCell>
                <TableCell className="text-slate-300">
                  {formatValue(row.Value_168h, row.parameter)}
                </TableCell>
                <TableCell className="text-slate-300">
                  {formatValue(row.predicted_168h, row.parameter)}
                </TableCell>
                <TableCell className="text-slate-300">
                  {formatZScore(row.robust_z_score)}
                </TableCell>
                <TableCell>{getStatusBadge(row)}</TableCell>
                <TableCell onClick={(e) => e.stopPropagation()}>
                  <JustificationPopover justification={row.justification} />
                </TableCell>
              </TableRow>
              {isExpanded && (
                <TableRow className="border-slate-800 bg-slate-950/80 hover:bg-slate-950/80">
                  <TableCell colSpan={10} className="py-4">
                    <div className="border border-amber-500/20 bg-amber-500/5 p-4">
                      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-amber-300">
                        Explainability — {row.ComponentID}
                      </p>

                      <div className="mb-3 flex items-center gap-2">
                        {getStatusBadge(row)}
                        <span className="text-xs text-slate-500">
                          {humanizeParameter(row.parameter)}
                        </span>
                      </div>
                      <p className="text-sm leading-relaxed text-slate-300">
                        {row.justification}
                      </p>
                      {row.reason_codes.length > 0 && (
                        <div className="mt-3 flex flex-wrap gap-2">
                          {row.reason_codes.map((code) => (
                            <Badge
                              key={code}
                              variant="outline"
                              className="border-slate-700 bg-slate-900/70 text-slate-300"
                            >
                              {code}
                            </Badge>
                          ))}
                        </div>
                      )}
                      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                        <div className="border border-slate-800 bg-slate-900/40 p-2.5">
                          <p className="text-[11px] uppercase tracking-wide text-slate-500">Isolation Forest</p>
                          <p className="mt-1 text-sm text-slate-200">
                            {formatZScore(row.isolation_forest_score)}
                          </p>
                        </div>
                        <div className="border border-slate-800 bg-slate-900/40 p-2.5">
                          <p className="text-[11px] uppercase tracking-wide text-slate-500">Predicted Drift</p>
                          <p className="mt-1 text-sm text-slate-200">
                            {formatValue(row.predicted_drift_rate, row.parameter, 4)} / h
                          </p>
                        </div>
                        <div className="border border-slate-800 bg-slate-900/40 p-2.5">
                          <p className="text-[11px] uppercase tracking-wide text-slate-500">Safety Slope</p>
                          <p className="mt-1 text-sm text-slate-200">
                            {formatValue(row.safety_slope, row.parameter, 4)} / h
                          </p>
                        </div>
                        <div className="border border-slate-800 bg-slate-900/40 p-2.5">
                          <p className="text-[11px] uppercase tracking-wide text-slate-500">Datasheet Limit</p>
                          <p className="mt-1 text-sm text-slate-200">
                            {formatLimit(row.datasheet_limit, row.parameter)}
                          </p>
                        </div>
                      </div>
                    </div>
                  </TableCell>
                </TableRow>
              )}
            </Fragment>
          )
        })}
      </TableBody>
    </Table>
  )
}

function JustificationPopover({ justification }) {
  return (
    <div className="group relative inline-flex">
      <button
        type="button"
        className="p-1 text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200"
        aria-label="View model justification"
      >
        <Info className="size-4" />
      </button>
      <div className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-2 hidden w-72 -translate-x-1/2 border border-slate-700 bg-slate-900 p-3 text-xs leading-relaxed text-slate-200 shadow-xl group-hover:block group-focus-within:block">
        <p className="mb-1 font-medium text-amber-300">Model Justification</p>
        <p>{justification}</p>
      </div>
    </div>
  )
}

function AccuracyMetricsTable({ rows, maeByParameter, filtered }) {
  const grouped = useMemo(() => {
    const groups = new Map()

    rows.forEach((row) => {
      if (!groups.has(row.parameter)) groups.set(row.parameter, [])
      groups.get(row.parameter).push(row)
    })

    return [...groups.entries()]
      .map(([parameter, parameterRows]) => {
        const evaluated = parameterRows.filter(
          (row) =>
            row.predicted_168h != null &&
            !Number.isNaN(Number(row.predicted_168h)) &&
            !Number.isNaN(Number(row.Value_168h)),
        )

        const clientMae = computeMae(parameterRows)
        const backendMae = toFiniteNumber(maeByParameter?.[parameter])
        const mae = filtered ? clientMae : backendMae ?? clientMae
        const flagged = parameterRows.filter(isFlagged).length

        return {
          parameter,
          samples: evaluated.length,
          records: parameterRows.length,
          flagged,
          mae,
        }
      })
      .sort((a, b) => a.parameter.localeCompare(b.parameter))
  }, [rows, maeByParameter, filtered])

  if (grouped.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-slate-500">
        No accuracy metrics available for the current view.
      </div>
    )
  }

  return (
    <div className="h-full overflow-auto border border-slate-800 bg-slate-950/40">
      <Table>
        <TableHeader className="sticky top-0 z-10 bg-slate-900">
          <TableRow className="border-slate-800 hover:bg-transparent">
            <TableHead className="text-slate-400">Parameter</TableHead>
            <TableHead className="text-slate-400">Evaluated Samples</TableHead>
            <TableHead className="text-slate-400">MAE</TableHead>
            <TableHead className="text-slate-400">Flagged</TableHead>
            <TableHead className="text-slate-400">Records</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {grouped.map((item) => (
            <TableRow key={item.parameter} className="border-slate-800">
              <TableCell className="font-medium text-slate-200">
                {humanizeParameter(item.parameter)}
              </TableCell>
              <TableCell className="text-slate-300">{item.samples}</TableCell>
              <TableCell className="text-slate-200">
                {item.mae != null
                  ? `${item.mae.toFixed(2)} ${parameterUnit(item.parameter)}`
                  : '—'}
              </TableCell>
              <TableCell className="text-slate-300">{item.flagged}</TableCell>
              <TableCell className="text-slate-300">{item.records}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

export default function App() {
  const fileInputRef = useRef(null)
  const [results, setResults] = useState(null)
  const [lotFilter, setLotFilter] = useState('all')
  const [parameterFilter, setParameterFilter] = useState('all')
  const [datasheetLimit, setDatasheetLimit] = useState(50)
  const [riskTolerance, setRiskTolerance] = useState(50)
  const [activeTab, setActiveTab] = useState('visualizer')
  const [expandedRowId, setExpandedRowId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [fileName, setFileName] = useState(null)
  const [uploadedFile, setUploadedFile] = useState(null)
  const analysisRequestRef = useRef(0)
  const [valueMode, setValueMode] = useState('percent') // 'raw' | 'percent' -- percent is the default: it's directly comparable across parameters with different absolute scales/limits

  const filteredRows = useMemo(() => {
    if (!results?.data) return []
    return results.data.filter((row) => {
      const lotMatch = lotFilter === 'all' || lotKey(row.Lot) === lotKey(lotFilter)
      const paramMatch = parameterFilter === 'all' || row.parameter === parameterFilter
      return lotMatch && paramMatch
    })
  }, [results, lotFilter, parameterFilter])

  const lots = useMemo(() => {
    if (!results?.data) return []
    return [...new Set(results.data.map((row) => lotKey(row.Lot)))]
      .filter((lot) => lot && lot !== '—')
      .sort()
  }, [results])

  const parameters = useMemo(() => {
    if (!results?.data) return []
    return [...new Set(results.data.map((row) => row.parameter))]
      .filter(Boolean)
      .sort()
  }, [results])

  // The datasheet limit to show as a single reference line/label: only
  // meaningful when the visible rows share one parameter (and therefore,
  // in practice, one limit). Take the first row's value rather than
  // assuming -- if rows disagree (shouldn't happen for one parameter, but
  // don't take it on faith), fall back to none rather than guess.
  const activeParameterLimit = useMemo(() => {
    if (parameterFilter === 'all' || filteredRows.length === 0) return null
    const limits = new Set(
      filteredRows.map((row) => row.datasheet_limit).filter((v) => v != null),
    )
    return limits.size === 1 ? [...limits][0] : null
  }, [filteredRows, parameterFilter])

  const componentCount = useMemo(() => {
    if (!results) return null
    if (parameterFilter === 'all' && lotFilter === 'all') {
      return results.components ?? new Set(filteredRows.map((r) => r.ComponentID)).size
    }
    // Filtered view: components can repeat across parameter rows, so count
    // distinct ComponentIDs actually in view rather than reusing the
    // unfiltered backend total.
    return new Set(filteredRows.map((r) => r.ComponentID)).size
  }, [results, filteredRows, parameterFilter, lotFilter])

  const mae = useMemo(() => computeMae(filteredRows), [filteredRows])
  const commonUnit = useMemo(() => getCommonUnit(filteredRows), [filteredRows])
  const accuracyTableUsesFilteredMae = lotFilter !== 'all' || parameterFilter !== 'all'

  const accuracyTableRows = useMemo(() => filteredRows, [filteredRows])

  const flaggedInView = useMemo(
    () => filteredRows.filter(isFlagged).length,
    [filteredRows],
  )

  const flaggedRate = useMemo(() => {
    if (filteredRows.length === 0) return null
    return (flaggedInView / filteredRows.length) * 100
  }, [filteredRows.length, flaggedInView])

  const chartOption = useMemo(
    () => buildChartOption(filteredRows, valueMode, activeParameterLimit),
    [filteredRows, valueMode, activeParameterLimit],
  )

  const parityOption = useMemo(
    () => buildParityChartOption(filteredRows),
    [filteredRows],
  )

  const analyzeFile = useCallback(async (file, nextRiskTolerance, nextDatasheetLimit) => {
    const requestId = ++analysisRequestRef.current
    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('risk_tolerance', String(nextRiskTolerance))
    formData.append('datasheet_limit', String(nextDatasheetLimit))

    try {
      const { data } = await axios.post(API_URL, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      if (requestId !== analysisRequestRef.current) return

      setResults(parseApiResponse(data))
      setExpandedRowId(null)
    } catch (err) {
      if (requestId !== analysisRequestRef.current) return

      const message =
        err.response?.data?.detail ??
        err.message ??
        'Failed to analyze CSV.'
      setError(message)
      setResults(null)
    } finally {
      if (requestId === analysisRequestRef.current) {
        setLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    if (!uploadedFile) return undefined

    const timeoutId = window.setTimeout(() => {
      analyzeFile(
        uploadedFile,
        riskTolerance,
        datasheetLimit,
      )
    }, 300)

    return () => window.clearTimeout(timeoutId)
  }, [uploadedFile, riskTolerance, analyzeFile])

  const handleUpload = useCallback((event) => {
    const file = event.target.files?.[0]
    if (!file) return

    analysisRequestRef.current += 1
    setLoading(false)
    setError(null)
    setFileName(file.name)
    setExpandedRowId(null)
    setLotFilter('all')
    setParameterFilter('all')
    setResults(null)
    setUploadedFile(file)
    event.target.value = ''
  }, [])

  const toggleRow = useCallback((rowKey) => {
    setExpandedRowId((current) => (
      current === rowKey ? null : rowKey
    ))
  }, [])

  return (
    <div className="dark flex h-full flex-col overflow-hidden bg-slate-950 text-slate-100">
      <div className="mx-auto flex h-full w-full max-w-[1600px] flex-col gap-3 overflow-hidden p-4 md:p-5">
        {/* Top Bar */}
        <header className="flex shrink-0 flex-col gap-3 border border-slate-800 bg-slate-900/60 p-3 md:flex-row md:items-end md:justify-between">
          <div className="min-w-0 space-y-0.5">
            <div className="flex items-center gap-2 text-amber-400">
              <Microscope className="size-4" />
              <span className="text-xs font-semibold uppercase tracking-widest">
                Burn-In Screening Pipeline
              </span>
            </div>
            <h1 className="text-xl font-semibold tracking-tight text-slate-50 md:text-2xl">
              AI-Driven Anomaly Detection Dashboard
            </h1>
            <p className="hidden max-w-2xl text-sm text-slate-400 lg:block">
              Module A: dynamic lot-relative outlier detection. Module B:
              time-series drift prediction from 0h/24h to forecast 168h
              behavior.
            </p>
          </div>

          <div className="flex flex-wrap items-end gap-2">
            <div className="space-y-1">
              <label
                htmlFor="lot-filter"
                className="text-xs font-medium text-slate-400"
              >
                Lot Filter
              </label>
              <div className="relative">
                <select
                  id="lot-filter"
                  value={lotFilter}
                  onChange={(e) => {
                    setLotFilter(e.target.value)
                    setExpandedRowId(null)
                  }}
                  disabled={!results}
                  className="h-9 min-w-[140px] appearance-none border border-slate-700 bg-slate-900 pr-8 pl-3 text-sm text-slate-200 outline-none focus:border-amber-500/60 focus:ring-2 focus:ring-amber-500/20 disabled:opacity-50"
                >
                  <option value="all">All Lots</option>
                  {lots.map((lot) => (
                    <option key={lot} value={lot}>
                      {lot}
                    </option>
                  ))}
                </select>
                <ChevronDown className="pointer-events-none absolute top-1/2 right-2 size-4 -translate-y-1/2 text-slate-500" />
              </div>
            </div>

            <div className="space-y-1">
              <label
                htmlFor="parameter-filter"
                className="text-xs font-medium text-slate-400"
              >
                Parameter
              </label>
              <div className="relative">
                <select
                  id="parameter-filter"
                  value={parameterFilter}
                  onChange={(e) => {
                    setParameterFilter(e.target.value)
                    setExpandedRowId(null)
                  }}
                  disabled={!results}
                  className="h-9 min-w-[160px] appearance-none border border-slate-700 bg-slate-900 pr-8 pl-3 text-sm text-slate-200 outline-none focus:border-amber-500/60 focus:ring-2 focus:ring-amber-500/20 disabled:opacity-50"
                >
                  <option value="all">All Parameters</option>
                  {parameters.map((param) => (
                    <option key={param} value={param}>
                      {humanizeParameter(param)}
                    </option>
                  ))}
                </select>
                <ChevronDown className="pointer-events-none absolute top-1/2 right-2 size-4 -translate-y-1/2 text-slate-500" />
              </div>
              {activeParameterLimit != null && (
                <p className="text-[11px] text-amber-400/80">
                  Datasheet limit: {formatLimit(activeParameterLimit, parameterFilter)}
                </p>
              )}
            </div>

            <div className="space-y-1">
              <label
                htmlFor="datasheet-limit"
                className="text-xs font-medium text-slate-400"
              >
                Datasheet Limit ({parameterFilter === 'all' ? 'native units' : parameterUnit(parameterFilter)})
              </label>
              <input
                id="datasheet-limit"
                type="number"
                min={0}
                step={0.1}
                value={datasheetLimit}
                onChange={(e) =>
                  setDatasheetLimit(Number(e.target.value) || 0)
                }
                className="h-9 w-[120px] border border-slate-700 bg-slate-900 px-3 text-sm text-slate-200 outline-none focus:border-amber-500/60 focus:ring-2 focus:ring-amber-500/20"
              />
            </div>

            <div className="space-y-1">
              <label
                htmlFor="risk-tolerance"
                className="text-xs font-medium text-slate-400"
              >
                Risk Tolerance (Cost Sensitivity)
              </label>
              <div className="flex items-center gap-2">
                <input
                  id="risk-tolerance"
                  type="range"
                  min={1}
                  max={100}
                  value={riskTolerance}
                  onChange={(e) => setRiskTolerance(Number(e.target.value))}
                  className="h-9 w-[120px] accent-amber-500"
                  title="Lower tolerance penalizes false negatives more"
                />
                <span className="text-xs text-slate-400">{riskTolerance}</span>
              </div>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={handleUpload}
            />
            <Button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              className="h-9 bg-amber-500 text-slate-950 hover:bg-amber-400"
            >
              {loading ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Upload className="size-4" />
              )}
              {loading ? 'Analyzing…' : 'Upload CSV'}
            </Button>
          </div>
        </header>

        {(fileName || error) && (
          <div className="shrink-0 space-y-2">
            {fileName && (
              <p className="text-xs text-slate-500">
                Last uploaded:{' '}
                <span className="text-slate-300">{fileName}</span>
              </p>
            )}
            {error && (
              <div className="flex items-start gap-3 border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-200">
                <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                <p>{error}</p>
              </div>
            )}
          </div>
        )}

        {/* Summary Cards */}
        <section className="grid shrink-0 gap-3 sm:grid-cols-2 xl:grid-cols-3">
          <SummaryCard
            title="Components"
            value={componentCount ?? '—'}
            description={
              results
                ? `${filteredRows.length} parametric record${filteredRows.length === 1 ? '' : 's'} in current view`
                : 'Screened physical parts (unique component IDs)'
            }
            icon={Activity}
            accent="bg-sky-500/15 text-sky-300"
          />
          <SummaryCard
            title="Flagged Rate"
            value={
              flaggedRate != null ? `${flaggedRate.toFixed(1)}%` : '—'
            }
            description={
              activeParameterLimit != null
                ? `${flaggedInView} flagged — vs. ${formatLimit(activeParameterLimit, parameterFilter)} limit (${humanizeParameter(parameterFilter)})`
                : `${flaggedInView} flagged — datasheet limit varies by parameter, select one to compare`
            }
            icon={AlertTriangle}
            accent="bg-red-500/15 text-red-300"
          />
          <SummaryCard
            title="Drift Prediction MAE"
            value={mae != null ? `${mae.toFixed(2)} ${commonUnit}` : '—'}
            description="Mean absolute error between predicted and actual 168h values (Module B)"
            icon={Target}
            accent="bg-amber-500/15 text-amber-300"
          />
        </section>

        {/* Tabs */}
        <div className="flex shrink-0 items-center gap-2 border-b border-slate-800 pb-2">
          <TabButton
            active={activeTab === 'visualizer'}
            onClick={() => setActiveTab('visualizer')}
            icon={LineChart}
            label="Trajectory Visualizer"
          />
          <TabButton
            active={activeTab === 'accuracy'}
            onClick={() => setActiveTab('accuracy')}
            icon={Crosshair}
            label="Prediction Accuracy"
          />
          <TabButton
            active={activeTab === 'registry'}
            onClick={() => setActiveTab('registry')}
            icon={TableProperties}
            label="Component Registry"
            count={results ? filteredRows.length : null}
          />
        </div>

        {/* Tab Panels */}
        <main className="min-h-0 flex-1 overflow-hidden">
          {activeTab === 'visualizer' ? (
            <Card className="flex h-full flex-col border-slate-800 bg-slate-900/70 ring-slate-800">
              <CardHeader className="shrink-0 flex-row items-start justify-between gap-3 pb-2">
                <div>
                  <CardTitle className="text-slate-50">
                    Parametric Trajectory Visualizer
                  </CardTitle>
                  <CardDescription className="text-slate-400">
                    {valueMode === 'percent' ? (
                      <>
                        Each trace shown as % of that component's own datasheet
                        limit, so tiers with different absolute scales (µA)
                        are directly comparable. 100% = at limit.
                      </>
                    ) : (
                      <>
                        Faint slate traces for passing components; bold
                        red/amber for flagged items. Dashed segments show Module B predicted 168h using only 0h/24h measurements.
                      </>
                    )}
                  </CardDescription>
                </div>
                <div className="flex shrink-0 border border-slate-700">
                  <button
                    type="button"
                    onClick={() => setValueMode('raw')}
                    className={`h-8 px-3 text-xs font-medium transition-colors ${
                      valueMode === 'raw'
                        ? 'bg-amber-500/15 text-amber-300'
                        : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                    }`}
                  >
                    Raw µA
                  </button>
                  <button
                    type="button"
                    onClick={() => setValueMode('percent')}
                    className={`h-8 border-l border-slate-700 px-3 text-xs font-medium transition-colors ${
                      valueMode === 'percent'
                        ? 'bg-amber-500/15 text-amber-300'
                        : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                    }`}
                  >
                    % of Limit
                  </button>
                </div>
              </CardHeader>
              <CardContent className="min-h-0 flex-1 pb-4">
                {loading ? (
                  <div className="flex h-full min-h-[200px] flex-col items-center justify-center gap-4 border border-dashed border-slate-800 bg-slate-950/50">
                    <BouncingDots />
                    <p className="text-sm font-medium text-amber-500/80 animate-pulse">
                      Running anomaly models and screening rules...
                    </p>
                  </div>
                ) : !results ? (
                  <div className="flex h-full min-h-[200px] flex-col items-center justify-center gap-3 border border-dashed border-slate-800 bg-slate-950/50 text-slate-500">
                    <Upload className="size-8 opacity-40" />
                    <p className="text-sm">
                      Upload a burn-in CSV to visualize parameter trajectories
                    </p>
                  </div>
                ) : (
                  <ReactECharts
                    option={chartOption}
                    style={{ height: '100%', width: '100%' }}
                    notMerge
                    lazyUpdate
                  />
                )}
              </CardContent>
            </Card>
          ) : activeTab === 'accuracy' ? (
            <Card className="flex h-full flex-col border-slate-800 bg-slate-900/70 ring-slate-800">
              <CardHeader className="shrink-0 pb-2">
                <CardTitle className="text-slate-50">
                  Predicted vs. Actual 168h (Module B)
                </CardTitle>
                <CardDescription className="text-slate-400">
                  Each point is one parametric record. Distance from the dashed
                  diagonal is the prediction error. Hover a point for its status,
                  reason, and deterministic justification.
                </CardDescription>
              </CardHeader>
              <CardContent className="min-h-0 flex-1 pb-4">
                {loading ? (
                  <div className="flex h-full min-h-[200px] flex-col items-center justify-center gap-4 border border-dashed border-slate-800 bg-slate-950/50">
                    <BouncingDots />
                    <p className="text-sm font-medium text-amber-500/80 animate-pulse">
                      Running anomaly models and screening rules...
                    </p>
                  </div>
                ) : !results ? (
                  <div className="flex h-full min-h-[200px] flex-col items-center justify-center gap-3 border border-dashed border-slate-800 bg-slate-950/50 text-slate-500">
                    <Crosshair className="size-8 opacity-40" />
                    <p className="text-sm">
                      Upload a burn-in CSV to compare predicted vs. actual 168h values
                    </p>
                  </div>
                ) : (
                  <div className="flex h-full min-h-0 flex-col gap-3">
                    <div className="min-h-[330px] h-[56%] shrink-0">
                      <ReactECharts
                        option={parityOption}
                        style={{ height: '100%', width: '100%' }}
                        notMerge
                        lazyUpdate
                      />
                    </div>
                    <div className="min-h-0 flex-1">
                      <AccuracyMetricsTable
                        rows={accuracyTableRows}
                        maeByParameter={results.mae_by_parameter}
                        filtered={accuracyTableUsesFilteredMae}
                      />
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          ) : (
            <Card className="flex h-full flex-col border-slate-800 bg-slate-900/70 ring-slate-800">
              <CardHeader className="shrink-0 pb-2">
                <CardTitle className="text-slate-50">
                  Component Registry
                </CardTitle>
                <CardDescription className="text-slate-400">
                  {lotFilter === 'all' && parameterFilter === 'all'
                    ? 'All lots, all parameters. Click a row to expand the model justification.'
                    : `${lotFilter === 'all' ? 'All lots' : `Lot ${lotFilter}`} · ${
                        parameterFilter === 'all' ? 'all parameters' : humanizeParameter(parameterFilter)
                      } — ${filteredRows.length} record${filteredRows.length === 1 ? '' : 's'} (${componentCount} component${componentCount === 1 ? '' : 's'}).`}
                </CardDescription>
              </CardHeader>
              <CardContent className="min-h-0 flex-1 overflow-hidden pb-4">
                <div className="h-full overflow-auto border border-slate-800 bg-slate-950/40">
                  <ComponentRegistry
                    key={`${lotFilter}-${parameterFilter}`}
                    rows={filteredRows}
                    expandedRowId={expandedRowId}
                    onToggleRow={toggleRow}
                    loading={loading}
                    hasResults={!!results}
                  />
                </div>
              </CardContent>
            </Card>
          )}
        </main>
      </div>
    </div>
  )
}