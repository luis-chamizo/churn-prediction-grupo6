// Renderiza un artículo en formato IEEE (dos columnas) a partir de un JSON de bloques.
// Uso: node render_ieee.js contenido.json salida.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, WidthType, BorderStyle, SectionType, ShadingType, LevelFormat,
  Footer, PageNumber, VerticalAlign, TableLayoutType,
} = require("docx");

const [, , inFile, outFile] = process.argv;
const spec = JSON.parse(fs.readFileSync(inFile, "utf8"));
const baseDir = path.dirname(path.resolve(inFile));

const FONT = "Times New Roman";
const PAGE_W = 12240, PAGE_H = 15840;                 // Carta
const M_SIDE = 900, M_TOP = 1080, M_BOTTOM = 1440;    // 0.625" / 0.75" / 1"
const COL_SPACE = 360;                                 // 0.25"
const COL_W = (PAGE_W - 2 * M_SIDE - COL_SPACE) / 2;   // ≈ 5040 dxa = 3.5"
const COL_W_PX = COL_W / 1440 * 96;

// ---------- utilidades ----------
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}

// Texto con marcado mínimo: **negrita**, *cursiva*, ^superíndice^
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), font: FONT, bold: true, ...base }));
    else if (t.startsWith("^")) out.push(new TextRun({ text: t.slice(1, -1), font: FONT, superScript: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), font: FONT, italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

const roman = n => ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"][n];
const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
let sec = 0, sub = 0, figN = 0, tabN = 0;

function body(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: opts.noIndent ? undefined : { firstLine: 202 },
    spacing: { after: 0, line: 228 },
    children: runs(text, { size: 20 }),
  });
}

function block(b) {
  switch (b.type) {
    case "h1": {
      sec += 1; sub = 0;
      return [new Paragraph({
        alignment: AlignmentType.CENTER, keepNext: true,
        spacing: { before: 200, after: 100 },
        children: [new TextRun({ text: `${roman(sec)}. ${b.text.toUpperCase()}`, font: FONT, size: 20, smallCaps: true })],
      })];
    }
    case "h1_nonum":
      return [new Paragraph({
        alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 100 },
        children: [new TextRun({ text: b.text.toUpperCase(), font: FONT, size: 20, smallCaps: true })],
      })];
    case "h2": {
      sub += 1;
      return [new Paragraph({
        keepNext: true, spacing: { before: 120, after: 60 },
        children: [new TextRun({ text: `${letters[sub - 1]}. ${b.text}`, font: FONT, size: 20, italics: true })],
      })];
    }
    case "p":
      return [body(b.text, b)];
    case "bullets":
      return b.items.map(t => new Paragraph({
        numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 0, line: 228 }, children: runs(t, { size: 20 }),
      }));
    case "numbered":
      return b.items.map(t => new Paragraph({
        numbering: { reference: "numbers-" + (b.ref || "a"), level: 0 }, alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 0, line: 228 }, children: runs(t, { size: 20 }),
      }));
    case "eq":
      return [new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { before: 60, after: 60 },
        children: [...runs(b.text, { size: 20, italics: false }),
          new TextRun({ text: `\t(${b.n})`, font: FONT, size: 20 })],
        tabStops: [{ type: "right", position: COL_W - 20 }],
      })];
    case "fig":
    case "widefig": {
      figN += 1;
      const f = path.resolve(baseDir, b.src);
      const { w, h, data } = pngSize(f);
      const fullW = (PAGE_W - 2 * M_SIDE) / 1440 * 96;
      const wpx = (b.type === "widefig" ? fullW : COL_W_PX) * (b.scale || 1);
      const hpx = wpx * h / w;
      return [
        new Paragraph({
          alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 40 },
          children: [new ImageRun({ type: "png", data, transformation: { width: Math.round(wpx), height: Math.round(hpx) },
            altText: { title: `Fig. ${figN}`, description: b.caption, name: `fig${figN}` } })],
        }),
        new Paragraph({
          alignment: AlignmentType.JUSTIFIED, spacing: { after: 140, line: 200 },
          children: [new TextRun({ text: `Fig. ${figN}. `, font: FONT, size: 16 }), ...runs(b.caption, { size: 16 })],
        }),
      ];
    }
    case "table": {
      tabN += 1;
      const ncol = b.rows[0].length;
      const widths = b.widths
        ? b.widths.map(x => Math.round(x * COL_W))
        : Array(ncol).fill(Math.floor(COL_W / ncol));
      const total = widths.reduce((a, c) => a + c, 0);
      const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
      const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
      const rows = b.rows.map((r, i) => new TableRow({
        tableHeader: i === 0, cantSplit: true,
        children: r.map((c, j) => new TableCell({
          width: { size: widths[j], type: WidthType.DXA },
          margins: { top: 30, bottom: 30, left: 60, right: 60 },
          verticalAlign: VerticalAlign.CENTER,
          shading: i === 0 ? { fill: "EDEDED", type: ShadingType.CLEAR, color: "auto" } : undefined,
          borders: { top: border, bottom: border, left: none, right: none },
          children: [new Paragraph({
            alignment: j === 0 && b.leftFirst !== false ? AlignmentType.LEFT : AlignmentType.CENTER,
            spacing: { after: 0, line: 200 },
            children: runs(String(c), { size: 15, bold: i === 0 || (b.boldRow === i) }),
          })],
        })),
      }));
      return [
        new Paragraph({
          alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 140, after: 0 },
          children: [new TextRun({ text: `TABLA ${roman(tabN)}`, font: FONT, size: 16 })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER, keepNext: true, spacing: { after: 60 },
          children: [new TextRun({ text: b.caption.toUpperCase(), font: FONT, size: 16, smallCaps: true })],
        }),
        new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows,
          alignment: AlignmentType.CENTER, layout: TableLayoutType.FIXED }),
        new Paragraph({ spacing: { after: 100 }, children: b.note ? runs(b.note, { size: 14, italics: true }) : [] }),
      ];
    }
    case "refs":
      return b.items.map((t, i) => new Paragraph({
        alignment: AlignmentType.LEFT, spacing: { after: 30, line: 200 },
        indent: { left: 360, hanging: 360 },
        children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 16 }), ...runs(t, { size: 16 })],
        tabStops: [{ type: "left", position: 360 }],
      }));
    default:
      throw new Error("Bloque desconocido: " + b.type);
  }
}

// ---------- cabecera (una columna) ----------
const header = [
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: [new TextRun({ text: spec.title, font: FONT, size: 44 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new TextRun({ text: spec.authors.join("     "), font: FONT, size: 22 })],
  }),
  ...spec.affiliation.map(a => new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 0 },
    children: [new TextRun({ text: a, font: FONT, size: 18, italics: true })],
  })),
  new Paragraph({ spacing: { after: 200 }, children: [] }),
];

// ---------- cuerpo (dos columnas) ----------
const abstract = [
  new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 100, line: 216 },
    children: [new TextRun({ text: "Resumen—", font: FONT, size: 18, bold: true, italics: true }),
      ...runs(spec.abstract, { size: 18, bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 140, line: 216 },
    children: [new TextRun({ text: "Palabras clave—", font: FONT, size: 18, bold: true, italics: true }),
      new TextRun({ text: spec.keywords, font: FONT, size: 18, bold: true })],
  }),
];
// Cada "widefig" cierra la sección de dos columnas, inserta una sección de una
// columna con la figura y abre de nuevo dos columnas (saltos de sección continuos).
const bodySections = [];
let current = [...abstract];
for (const b of spec.blocks) {
  if (b.type === "widefig") {
    bodySections.push({ cols: 2, children: current });
    bodySections.push({ cols: 1, children: block(b) });
    current = [];
  } else current.push(...block(b));
}
bodySections.push({ cols: 2, children: current });

const numberingRefs = ["a", "b", "c", "d"].map(r => ({
  reference: "numbers-" + r,
  levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1)", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 360, hanging: 260 } }, run: { font: FONT, size: 20 } } }],
}));

const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] });

const page = { size: { width: PAGE_W, height: PAGE_H },
  margin: { top: M_TOP, bottom: M_BOTTOM, left: M_SIDE, right: M_SIDE } };

const doc = new Document({
  creator: spec.authors.join(", "), title: spec.title, description: "Documento técnico IEEE",
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 300, hanging: 200 } }, run: { font: FONT } } }] },
    ...numberingRefs] },
  sections: [
    { properties: { page }, footers: { default: footer }, children: header },
    ...bodySections.filter(x => x.children.length).map(x => ({
      properties: { page, type: SectionType.CONTINUOUS,
        column: x.cols === 2 ? { count: 2, space: COL_SPACE, equalWidth: true } : { count: 1 } },
      footers: { default: footer }, children: x.children })),
  ],
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync(outFile, buf); console.log("OK", outFile, figN, "figuras,", tabN, "tablas"); });
