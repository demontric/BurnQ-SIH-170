import { Fragment, useCallback, useMemo, useRef, useState } from 'react'
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

function isFlagged(row) {
  return Boolean(row.is_anomaly || row.safety_slope_exceeded)
}

function lotKey(value) {
  if (value == null || value === '') return '—'
  return String(value).trim()
}

function normalizeRow(row) {
  return {
    ComponentID:
      row.ComponentID ?? row.component_id ?? row.part_id ?? row.PartID ?? '—',
    Lot: lotKey(row.Lot ?? row.lot_id ?? row.lot),
    Value_0h: Number(row.Value_0h ?? row.value_0h ?? 0),
    Value_24h: Number(row.Value_24h ?? row.value_24h ?? 0),
    Value_96h: Number(row.Value_96h ?? row.value_96h ?? 0),
    Value_168h: Number(row.Value_168h ?? row.value_168h ?? 0),
    predicted_168h:
      row.predicted_168h != null ? Number(row.predicted_168h) : null,
    robust_z_score:
      row.robust_z_score != null ? Number(row.robust_z_score) : null,
    datasheet_limit:
      row.datasheet_limit != null ? Number(row.datasheet_limit) : null,
    is_anomaly: Boolean(row.is_anomaly),
    safety_slope_exceeded: Boolean(row.safety_slope_exceeded),
    justification:
      row.justification ??
      (isFlagged({
        is_anomaly: row.is_anomaly,
        safety_slope_exceeded: row.safety_slope_exceeded,
      })
        ? 'Flagged due to abnormal drift or statistical deviation.'
        : 'Normal part.'),
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
    total_components: payload?.total_components ?? data.length,
    flagged_count,
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

function formatMicroamps(value) {
  if (value == null || Number.isNaN(value)) return '—'
  return `${value.toFixed(2)} µA`
}

function formatZScore(value) {
  if (value == null || Number.isNaN(value)) return '—'
  return value.toFixed(2)
}

function getStatusBadge(row) {
  if (row.is_anomaly && row.safety_slope_exceeded) {
    return (
      <Badge variant="destructive" className="bg-red-500/20 text-red-300">
        Anomaly + Drift
      </Badge>
    )
  }
  if (row.is_anomaly) {
    return (
      <Badge variant="destructive" className="bg-red-500/20 text-red-300">
        Outlier
      </Badge>
    )
  }
  if (row.safety_slope_exceeded) {
    return (
      <Badge
        variant="outline"
        className="border-amber-500/50 bg-amber-500/15 text-amber-300"
      >
        Drift Risk
      </Badge>
    )
  }
  return (
    <Badge variant="secondary" className="bg-slate-800 text-slate-300">
      Pass
    </Badge>
  )
}

function getFlaggedColor(row) {
  if (row.is_anomaly) return '#ef4444'
  if (row.safety_slope_exceeded) return '#f59e0b'
  return '#94a3b8'
}

function buildChartOption(rows, datasheetLimit, mode = 'raw') {
  const series = []
  const legendEntries = []
  const isPercent = mode === 'percent'

  const scaleFor = (row) => {
    if (!isPercent) return 1
    const limit = row.datasheet_limit ?? datasheetLimit
    return limit > 0 ? 100 / limit : 1
  }

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
      legendEntries.push(`${row.ComponentID} (predicted)`)
    }

    if (flagged) legendEntries.push(row.ComponentID)
  })

  const limitLine = isPercent
    ? { value: 100, label: 'Datasheet Limit (100%)' }
    : { value: datasheetLimit, label: `Datasheet Limit (${datasheetLimit} µA)` }

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
        if (!params?.seriesName) return ''
        const label = params.name || TIME_LABELS[params.dataIndex] || ''
        const value =
          params.value != null && !Number.isNaN(params.value)
            ? isPercent
              ? `${Number(params.value).toFixed(1)}% of limit`
              : `${Number(params.value).toFixed(2)} µA`
            : '—'
        return `<strong>${params.seriesName.replace(' (predicted)', '')}</strong><br/>${label}: ${value}`
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
      name: isPercent ? '% of Datasheet Limit' : 'Parameter Value (µA)',
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
    ],
  }
}

function buildParityChartOption(rows) {
  const points = rows.filter(
    (row) =>
      row.predicted_168h != null &&
      !Number.isNaN(row.predicted_168h) &&
      !Number.isNaN(row.Value_168h),
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
  const min = Math.min(...allValues)
  const max = Math.max(...allValues)
  const pad = (max - min) * 0.05 || 1
  const lo = Math.max(0, min - pad)
  const hi = max + pad

  const flaggedPoints = []
  const passPoints = []

  points.forEach((row) => {
    const point = {
      value: [row.Value_168h, row.predicted_168h],
      name: row.ComponentID,
      error: row.predicted_168h - row.Value_168h,
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
        if (!params?.data || params.seriesType !== 'scatter') return ''
        const [actual, predicted] = params.data.value
        const error = params.data.error
        return (
          `<strong>${params.data.name}</strong><br/>` +
          `Actual 168h: ${Number(actual).toFixed(2)} µA<br/>` +
          `Predicted 168h: ${Number(predicted).toFixed(2)} µA<br/>` +
          `Error: ${error >= 0 ? '+' : ''}${Number(error).toFixed(2)} µA`
        )
      },
    },
    xAxis: {
      type: 'value',
      name: 'Actual Value_168h (µA)',
      nameLocation: 'middle',
      nameGap: 32,
      nameTextStyle: { color: '#94a3b8' },
      min: lo,
      max: hi,
      axisLine: { lineStyle: { color: '#475569' } },
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#1e293b', type: 'dashed' } },
    },
    yAxis: {
      type: 'value',
      name: 'Predicted 168h (µA)',
      nameTextStyle: { color: '#94a3b8', padding: [0, 0, 0, 8] },
      min: lo,
      max: hi,
      axisLine: { show: false },
      axisLabel: { color: '#94a3b8' },
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
          {count}
        </span>
      )}
    </button>
  )
}

// Bouncing Dots Component for LLM generation loading state
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
  explanations,
  rowLoading,
}) {
  if (loading) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 py-20">
        <BouncingDots />
        <p className="text-sm font-medium text-amber-500/80 animate-pulse">
          Generating LLM explanations & running anomaly models...
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
                  {formatMicroamps(row.Value_0h)}
                </TableCell>
                <TableCell className="text-slate-300">
                  {formatMicroamps(row.Value_24h)}
                </TableCell>
                <TableCell className="text-slate-300">
                  {formatMicroamps(row.Value_96h)}
                </TableCell>
                <TableCell className="text-slate-300">
                  {formatMicroamps(row.Value_168h)}
                </TableCell>
                <TableCell className="text-slate-300">
                  {formatMicroamps(row.predicted_168h)}
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

                      {rowLoading[rowKey] ? (
                        <div className="flex items-center gap-3 py-2">
                          <BouncingDots />
                          <span className="text-sm text-amber-500/80">
                            Analyzing component trajectory...
                          </span>
                        </div>
                      ) : (
                        <p className="text-sm leading-relaxed text-slate-300">
                          {explanations[rowKey] || 'No justification available.'}
                        </p>
                      )}
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

export default function App() {
  const fileInputRef = useRef(null)
  const [results, setResults] = useState(null)
  const [lotFilter, setLotFilter] = useState('all')
  const [datasheetLimit, setDatasheetLimit] = useState(50)
  const [riskTolerance, setRiskTolerance] = useState(50)
  const [activeTab, setActiveTab] = useState('visualizer')
  const [expandedRowId, setExpandedRowId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [fileName, setFileName] = useState(null)
  const [explanations, setExplanations] = useState({})
  const [rowLoading, setRowLoading] = useState({})
  const [valueMode, setValueMode] = useState('raw') // 'raw' | 'percent'

  const filteredRows = useMemo(() => {
    if (!results?.data) return []
    if (lotFilter === 'all') return results.data
    const selectedLot = lotKey(lotFilter)
    return results.data.filter((row) => lotKey(row.Lot) === selectedLot)
  }, [results, lotFilter])

  const lots = useMemo(() => {
    if (!results?.data) return []
    return [...new Set(results.data.map((row) => lotKey(row.Lot)))]
      .filter((lot) => lot && lot !== '—')
      .sort()
  }, [results])

  const mae = useMemo(() => computeMae(filteredRows), [filteredRows])

  const flaggedInView = useMemo(
    () => filteredRows.filter(isFlagged).length,
    [filteredRows],
  )

  const flaggedRate = useMemo(() => {
    if (filteredRows.length === 0) return null
    return (flaggedInView / filteredRows.length) * 100
  }, [filteredRows.length, flaggedInView])

  const chartOption = useMemo(
    () => buildChartOption(filteredRows, datasheetLimit, valueMode),
    [filteredRows, datasheetLimit, valueMode],
  )

  const parityOption = useMemo(
    () => buildParityChartOption(filteredRows),
    [filteredRows],
  )

  const handleUpload = useCallback(async (event) => {
    const file = event.target.files?.[0]
    if (!file) return

    setLoading(true)
    setError(null)
    setFileName(file.name)
    setExpandedRowId(null)
    setLotFilter('all')
    setResults(null) // Clear old results instantly while loading
    setExplanations({}) // Cached explanations are keyed by row, so drop them on new data
    setRowLoading({})

    const formData = new FormData()
    formData.append('file', file)
    formData.append('risk_tolerance', riskTolerance)
    formData.append('datasheet_limit', datasheetLimit)

    try {
      const { data } = await axios.post(API_URL, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResults(parseApiResponse(data))
    } catch (err) {
      const message =
        err.response?.data?.detail ??
        err.message ??
        'Failed to analyze CSV. Ensure the API is running on port 8000.'
      setError(message)
      setResults(null)
    } finally {
      setLoading(false)
      event.target.value = ''
    }
  }, [riskTolerance, datasheetLimit])

  const toggleRow = useCallback(async (rowKey, rowData) => {
    setExpandedRowId((current) => (current === rowKey ? null : rowKey))

    // If we already have the explanation, don't fetch it again
    if (explanations[rowKey] || !rowData) return

    setRowLoading((prev) => ({ ...prev, [rowKey]: true }))

    try {
      const response = await axios.post(
        'http://127.0.0.1:8000/api/explain',
        rowData,
      )
      setExplanations((prev) => ({
        ...prev,
        [rowKey]: response.data.justification,
      }))
    } catch (err) {
      setExplanations((prev) => ({
        ...prev,
        [rowKey]: 'Error generating explanation.',
      }))
    } finally {
      setRowLoading((prev) => ({ ...prev, [rowKey]: false }))
    }
  }, [explanations])

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
                htmlFor="datasheet-limit"
                className="text-xs font-medium text-slate-400"
              >
                Datasheet Limit (µA)
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
            title="Total Components"
            value={results ? filteredRows.length : '—'}
            description="Screened parametric burn-in trajectories in current view"
            icon={Activity}
            accent="bg-sky-500/15 text-sky-300"
          />
          <SummaryCard
            title="Flagged Rate"
            value={
              flaggedRate != null ? `${flaggedRate.toFixed(1)}%` : '—'
            }
            description={`${flaggedInView} flagged — monitor false negatives vs. static ${datasheetLimit} µA limit`}
            icon={AlertTriangle}
            accent="bg-red-500/15 text-red-300"
          />
          <SummaryCard
            title="Drift Prediction MAE"
            value={mae != null ? `${mae.toFixed(2)} µA` : '—'}
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
                        red/amber for flagged items. Dashed segments show
                        Module B predicted 168h from the 96h measurement.
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
                      Generating LLM explanations & running anomaly models...
                    </p>
                  </div>
                ) : !results ? (
                  <div className="flex h-full min-h-[200px] flex-col items-center justify-center gap-3 border border-dashed border-slate-800 bg-slate-950/50 text-slate-500">
                    <Upload className="size-8 opacity-40" />
                    <p className="text-sm">
                      Upload a burn-in CSV to visualize leakage-current
                      trajectories
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
                  Each point is one component. Distance from the dashed
                  diagonal is the prediction error — points above the line are
                  over-predicted, below are under-predicted. Flagged
                  components are colored red/amber; passing components are
                  faint slate.
                </CardDescription>
              </CardHeader>
              <CardContent className="min-h-0 flex-1 pb-4">
                {loading ? (
                  <div className="flex h-full min-h-[200px] flex-col items-center justify-center gap-4 border border-dashed border-slate-800 bg-slate-950/50">
                    <BouncingDots />
                    <p className="text-sm font-medium text-amber-500/80 animate-pulse">
                      Generating LLM explanations & running anomaly models...
                    </p>
                  </div>
                ) : !results ? (
                  <div className="flex h-full min-h-[200px] flex-col items-center justify-center gap-3 border border-dashed border-slate-800 bg-slate-950/50 text-slate-500">
                    <Crosshair className="size-8 opacity-40" />
                    <p className="text-sm">
                      Upload a burn-in CSV to compare predicted vs. actual
                      168h values
                    </p>
                  </div>
                ) : (
                  <ReactECharts
                    option={parityOption}
                    style={{ height: '100%', width: '100%' }}
                    notMerge
                    lazyUpdate
                  />
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
                  {lotFilter === 'all'
                    ? 'All lots. Click a row to expand the model justification.'
                    : `Filtered to lot ${lotFilter} — ${filteredRows.length} component${filteredRows.length === 1 ? '' : 's'}.`}
                </CardDescription>
              </CardHeader>
              <CardContent className="min-h-0 flex-1 overflow-hidden pb-4">
                <div className="h-full overflow-auto border border-slate-800 bg-slate-950/40">
                  <ComponentRegistry
                    key={lotFilter}
                    rows={filteredRows}
                    expandedRowId={expandedRowId}
                    onToggleRow={toggleRow}
                    loading={loading}
                    hasResults={!!results}
                    explanations={explanations}
                    rowLoading={rowLoading}
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