% MDPI Article Class - Simplified Version
% For Big Data and Cognitive Computing (BDCC) Journal
% Based on MDPI template structure

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Definitions/mdpi}[2024/12/23 MDPI Article Class (Simplified)]

% Options
\newif\if@submit\@submittrue
\newif\if@moreauthors\@moreauthorsfalse
\newif\if@pdftex\@pdftextrue

\DeclareOption{submit}{\@submittrue}
\DeclareOption{moreauthors}{\@moreauthorstrue}
\DeclareOption{pdftex}{\@pdftextrue}
\DeclareOption{journal}{}
\DeclareOption{article}{}

\ProcessOptions\relax

% Base class
\LoadClass[11pt,a4paper]{article}

% Required packages (minimal dependencies)
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{geometry}
\RequirePackage{setspace}
\RequirePackage{xcolor}
\RequirePackage{etoolbox}
\RequirePackage{pifont}

% Page geometry (MDPI style)
\geometry{
  left=2.5cm,
  right=2.5cm,
  top=2.5cm,
  bottom=2.5cm
}

% 1.5 line spacing
\onehalfspacing

% Journal information commands
\newcommand{\journal}[1]{\gdef\@journal{#1}}
\newcommand{\volume}[1]{\gdef\@volume{#1}}
\newcommand{\issue}[1]{\gdef\@issue{#1}}
\newcommand{\articlenumber}[1]{\gdef\@articlenumber{#1}}
\newcommand{\pubvolume}[1]{\gdef\@pubvolume{#1}}
\newcommand{\pubyear}[1]{\gdef\@pubyear{#1}}
\newcommand{\copyrightyear}[1]{\gdef\@copyrightyear{#1}}
\newcommand{\datereceived}[1]{\gdef\@datereceived{#1}}
\newcommand{\dateaccepted}[1]{\gdef\@dateaccepted{#1}}
\newcommand{\datepublished}[1]{\gdef\@datepublished{#1}}
\newcommand{\hreflink}[1]{\gdef\@hreflink{#1}}

% Author commands
\newcommand{\Title}[1]{\gdef\@Title{#1}\title{#1}}
\newcommand{\Author}[1]{\gdef\@Author{#1}\author{#1}}
\renewcommand{\abstract}[1]{\gdef\@abstract{#1}}
\newcommand{\keyword}[1]{\gdef\@keyword{#1}}
\newcommand{\corres}[1]{\gdef\@corres{#1}}
\newcommand{\firstnote}[1]{\gdef\@firstnote{#1}}
\newcommand{\address}[1]{\gdef\@address{#1}}

% Default values
\journal{Big Data Cogn. Comput.}
\volume{1}
\issue{1}
\articlenumber{0}
\pubvolume{1}
\pubyear{2025}
\copyrightyear{2025}

% Section formatting (MDPI style - no indentation)
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}

% Section numbering
\setcounter{secnumdepth}{4}

% Section titles (MDPI style)
\renewcommand{\section}{\@startsection{section}{1}{\z@}%
  {-12pt \@plus -4pt \@minus -2pt}%
  {6pt \@plus 2pt}%
  {\normalfont\large\bfseries}}

\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}%
  {-10pt \@plus -3pt \@minus -2pt}%
  {4pt \@plus 2pt}%
  {\normalfont\normalsize\bfseries}}

\renewcommand{\subsubsection}{\@startsection{subsubsection}{3}{\z@}%
  {-8pt \@plus -2pt \@minus -1pt}%
  {2pt \@plus 1pt}%
  {\normalfont\normalsize\itshape}}

% Title page
\renewcommand{\maketitle}{%
  \begin{center}
    {\Large\bfseries\@Title\par}
    \vspace{1em}
    {\normalsize\@Author\par}
    \vspace{0.5em}
    {\small\@address\par}
    \vspace{0.5em}
    {\small *Correspondence: \@corres\par}
    \vspace{1.5em}
    \hrule
    \vspace{1em}
    {\small\textbf{Abstract:} \@abstract\par}
    \vspace{1em}
    {\small\textbf{Keywords:} \@keyword\par}
    \vspace{1em}
    \hrule
  \end{center}
  \vspace{1em}
}

% Back matter sections (MDPI required)
\newcommand{\authorcontributions}[1]{%
  \vspace{1em}
  \noindent\textbf{Author Contributions:} #1
}

\newcommand{\funding}[1]{%
  \vspace{0.5em}
  \noindent\textbf{Funding:} #1
}

\newcommand{\dataavailability}[1]{%
  \vspace{0.5em}
  \noindent\textbf{Data Availability Statement:} #1
}

\newcommand{\acknowledgments}[1]{%
  \vspace{0.5em}
  \noindent\textbf{Acknowledgments:} #1
}

\newcommand{\conflictsofinterest}[1]{%
  \vspace{0.5em}
  \noindent\textbf{Conflicts of Interest:} #1
}

% Environment for adjustwidth
\newlength{\extralength}
\setlength{\extralength}{1cm}

% Adjustwidth environment (simplified)
\newenvironment{adjustwidth}[2]{%
  \begin{list}{}{%
    \setlength{\leftmargin}{#1}%
    \setlength{\rightmargin}{#2}%
  }\item[]%
}{\end{list}}

% Reference title
\newcommand{\reftitle}[1]{\vspace{1em}\noindent\textbf{#1}\vspace{0.5em}}

% Endnotes (simplified)
\newcommand{\printendnotes}[1]{}

% Float placement
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.75}

\endinput
