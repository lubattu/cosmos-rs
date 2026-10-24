//! Self-contained HTML page with an inline SVG chart.

use std::fmt::Write as _;

use super::exp_bench::BenchResults;

const WIDTH: f64 = 720.0;
const HEIGHT: f64 = 440.0;
const MARGIN: f64 = 64.0;
const SERIES_COLOURS: [&str; 5] = ["#1b3a6b", "#a33b20", "#2f6b34", "#6b3f8a", "#8a6b1f"];

fn axis_scale(values: &[f64]) -> (f64, f64) {
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (high - low).abs() < f64::EPSILON {
        (low - 1.0, high + 1.0)
    } else {
        (low, high)
    }
}

/// Render the log-log chart as an SVG fragment.
pub fn render_svg(results: &BenchResults) -> String {
    let mut series: Vec<(String, Vec<(f64, f64)>)> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    for entry in &results.methods {
        if !names.contains(&entry.method) {
            names.push(entry.method.clone());
        }
    }
    for name in &names {
        let points: Vec<(f64, f64)> = results
            .methods
            .iter()
            .filter(|entry| &entry.method == name && entry.median_s > 0.0)
            .map(|entry| ((entry.n as f64).log10(), entry.median_s.log10()))
            .collect();
        series.push((name.clone(), points));
    }
    series.push((
        "block qr".to_string(),
        results
            .block_qr
            .iter()
            .filter(|entry| entry.median_s > 0.0)
            .map(|entry| ((entry.n as f64).log10(), entry.median_s.log10()))
            .collect(),
    ));

    let xs: Vec<f64> = series
        .iter()
        .flat_map(|(_, points)| points.iter().map(|(x, _)| *x))
        .collect();
    let ys: Vec<f64> = series
        .iter()
        .flat_map(|(_, points)| points.iter().map(|(_, y)| *y))
        .collect();
    let (x_low, x_high) = axis_scale(&xs);
    let (y_low, y_high) = axis_scale(&ys);

    let to_px = |x: f64, y: f64| {
        let px = MARGIN + (x - x_low) / (x_high - x_low) * (WIDTH - 2.0 * MARGIN);
        let py = HEIGHT - MARGIN - (y - y_low) / (y_high - y_low) * (HEIGHT - 2.0 * MARGIN);
        (px, py)
    };

    let mut svg = String::new();
    let _ = write!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" role="img">"#
    );
    let _ = write!(
        svg,
        r##"<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>"##
    );
    let _ = write!(
        svg,
        r##"<g stroke="#333333" stroke-width="1" fill="none"><path d="M {MARGIN} {} L {MARGIN} {MARGIN} M {MARGIN} {} L {} {}"/></g>"##,
        HEIGHT - MARGIN,
        HEIGHT - MARGIN,
        WIDTH - MARGIN,
        HEIGHT - MARGIN
    );
    let _ = write!(
        svg,
        r#"<text x="{}" y="{}" font-family="sans-serif" font-size="13" text-anchor="middle">log10 ambient dimension</text>"#,
        WIDTH / 2.0,
        HEIGHT - 18.0
    );
    let _ = write!(
        svg,
        r#"<text x="18" y="{}" font-family="sans-serif" font-size="13" text-anchor="middle" transform="rotate(-90 18 {})">log10 median seconds</text>"#,
        HEIGHT / 2.0,
        HEIGHT / 2.0
    );

    for (index, (name, points)) in series.iter().enumerate() {
        if points.is_empty() {
            continue;
        }
        let colour = SERIES_COLOURS[index % SERIES_COLOURS.len()];
        let mut path = String::new();
        for (order, (x, y)) in points.iter().enumerate() {
            let (px, py) = to_px(*x, *y);
            let _ = write!(
                path,
                "{} {px:.2} {py:.2}",
                if order == 0 { "M" } else { " L" }
            );
        }
        let _ = write!(
            svg,
            r#"<path d="{path}" fill="none" stroke="{colour}" stroke-width="2"/>"#
        );
        for (x, y) in points {
            let (px, py) = to_px(*x, *y);
            let _ = write!(
                svg,
                r#"<circle cx="{px:.2}" cy="{py:.2}" r="3" fill="{colour}"/>"#
            );
        }
        let _ = write!(
            svg,
            r#"<text x="{}" y="{}" font-family="sans-serif" font-size="12" fill="{colour}">{name}</text>"#,
            WIDTH - MARGIN + 6.0,
            MARGIN + 16.0 * index as f64
        );
    }
    svg.push_str("</svg>");
    svg
}

/// Render the complete report page.
pub fn render_html(results: &BenchResults) -> String {
    let mut rows = String::new();
    for entry in &results.methods {
        let deviation = entry
            .frob_dev_from_pade
            .map(|value| format!("{value:.3e}"))
            .unwrap_or_else(|| "reference".to_string());
        let _ = write!(
            rows,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.6e}</td><td>{:.6e}</td><td>{:.3e}</td><td>{deviation}</td></tr>",
            entry.method,
            entry.n,
            entry.reps,
            entry.median_s,
            entry.min_s,
            entry.orthogonality_error
        );
    }
    let mut block_rows = String::new();
    for entry in &results.block_qr {
        let _ = write!(
            block_rows,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.6e}</td></tr>",
            entry.n, entry.k, entry.reps, entry.median_s
        );
    }
    let mut slope_rows = String::new();
    for (name, slope) in &results.slopes {
        let _ = write!(slope_rows, "<tr><td>{name}</td><td>{slope:.3}</td></tr>");
    }

    format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Orthogonal update cost</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; color: #1a1a1a; max-width: 60rem; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #bbbbbb; padding: 0.3rem 0.6rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
caption {{ text-align: left; font-weight: 600; padding-bottom: 0.4rem; }}
</style>
</head>
<body>
<h1>Orthogonal update cost</h1>
<p>Every generator is antisymmetric with Frobenius norm {target_norm}. Timings are
single-threaded wall clock; each pair runs one untimed warm-up call before the
timed repetitions, and the median is reported.</p>
{svg}
<table><caption>Full-dimension updates</caption>
<tr><th>method</th><th>n</th><th>reps</th><th>median s</th><th>min s</th><th>orthogonality error</th><th>deviation</th></tr>
{rows}
</table>
<table><caption>Per-block orthonormalization</caption>
<tr><th>n</th><th>k</th><th>reps</th><th>median s</th></tr>
{block_rows}
</table>
<table><caption>Fitted log-log slopes</caption>
<tr><th>series</th><th>exponent</th></tr>
{slope_rows}
</table>
</body>
</html>
"#,
        target_norm = results.target_norm,
        svg = render_svg(results),
    )
}

#[cfg(test)]
mod tests {
    use super::super::exp_bench::measure;
    use super::*;

    #[test]
    fn page_is_self_contained() {
        let methods = vec!["pade".to_string(), "qr".to_string()];
        let results = measure(&[16, 32], &methods, 1.2, 8);
        let html = render_html(&results);
        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("<svg"));
        assert!(!html.contains("<script"));
        assert!(!html.contains("src="));
        assert!(!html.contains("<link"));
    }

    #[test]
    fn chart_draws_one_series_per_method_plus_blocks_and_the_axis() {
        let methods = vec!["pade".to_string(), "qr".to_string()];
        let results = measure(&[16, 32], &methods, 1.2, 8);
        let svg = render_svg(&results);
        assert_eq!(svg.matches("<path d=\"M").count(), 4);
    }
}
