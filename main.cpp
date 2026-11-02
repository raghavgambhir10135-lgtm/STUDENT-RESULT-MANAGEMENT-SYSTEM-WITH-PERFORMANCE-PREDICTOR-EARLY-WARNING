// UNIQUE FEATURE: Performance Predictor & Early-Warning System (Simple Linear Regression)
#include <iomanip>
#include <map>
#include "Feature.h"
#include "Utils.h"
 
namespace {
 
enum class Risk { LOW, MEDIUM, HIGH };
 
const char* riskName(Risk r) {
    switch (r) { case Risk::HIGH: return "HIGH"; case Risk::MEDIUM: return "MEDIUM"; default: return "LOW"; }
}
 
Risk classify(double predictedTotal) {
    if (predictedTotal < GradeScale::PASS_PERCENT) return Risk::HIGH;
    if (predictedTotal < GradeScale::PASS_PERCENT + 10) return Risk::MEDIUM;
    return Risk::LOW;
}
 
// y = slope * x + intercept, fitted by ordinary least squares.
struct RegressionModel {
    double slope = 0, intercept = 0, r2 = 0;
    size_t n = 0;
 
    double predict(double x) const {
        double y = slope * x + intercept;
        return std::max(0.0, std::min(SubjectMark::MAX_EXTERNAL, y));   // clamp to valid range
    }
 
    static RegressionModel fit(const std::vector<double>& x, const std::vector<double>& y) {
        RegressionModel m;
        m.n = x.size();
        if (m.n < 3) throw ResultException("At least 3 students are needed to train the predictor");
        double mx = util::mean(x), my = util::mean(y), sxx = 0, sxy = 0, syy = 0;
        for (size_t i = 0; i < m.n; ++i) {
            sxx += (x[i] - mx) * (x[i] - mx);
            sxy += (x[i] - mx) * (y[i] - my);
            syy += (y[i] - my) * (y[i] - my);
        }
        m.slope = sxx > 0 ? sxy / sxx : 0.0;
        m.intercept = my - m.slope * mx;
        m.r2 = (sxx > 0 && syy > 0) ? (sxy * sxy) / (sxx * syy) : 0.0;
        return m;
    }
};
 
class PerformancePredictor {
    const ResultManager& rm_;
    std::map<std::string, RegressionModel> models_;
 
public:
    explicit PerformancePredictor(const ResultManager& rm) : rm_(rm) { train(); }
 
    void train() {
        models_.clear();
        for (const SubjectMark& sub : ResultManager::catalogue()) {
            std::vector<double> in, ex;
            for (const Student& s : rm_.all())
                for (const SubjectMark& m : s.marks())
                    if (m.code == sub.code) { in.push_back(m.internal); ex.push_back(m.external); }
            models_[sub.code] = RegressionModel::fit(in, ex);
        }
    }
 
    const RegressionModel& model(const std::string& code) const { return models_.at(code); }
 
    // Mean absolute error of the model on the training data (back-test).
    double mae(const std::string& code) const {
        std::vector<double> err;
        for (const Student& s : rm_.all())
            for (const SubjectMark& m : s.marks())
                if (m.code == code) err.push_back(std::abs(model(code).predict(m.internal) - m.external));
        return util::mean(err);
    }
 
    void printModels(std::ostream& os) const {
        os << std::fixed << std::setprecision(3) << std::left << std::setw(9) << "Code" << std::right
           << std::setw(9) << "Slope" << std::setw(11) << "Intercept" << std::setw(8) << "R^2"
           << std::setw(8) << "MAE" << std::setw(5) << "n" << "\n" << util::line('-', 50) << "\n";
        for (const SubjectMark& sub : ResultManager::catalogue()) {
            const auto& m = model(sub.code);
            os << std::left << std::setw(9) << sub.code << std::right << std::setw(9) << m.slope
               << std::setw(11) << m.intercept << std::setw(8) << m.r2 << std::setw(8) << mae(sub.code)
               << std::setw(5) << m.n << "\n";
        }
    }
 
    // Back-test: predict every student's result from internal marks alone.
    void printEarlyWarning(std::ostream& os) const {
        os << std::left << std::setw(12) << "UID" << std::setw(18) << "Name" << std::right << std::setw(8)
           << "PredSGPA" << std::setw(8) << "ActSGPA" << std::setw(9) << "Risk" << "  Weakest subject\n"
           << util::line('-', 78) << "\n" << std::fixed << std::setprecision(2);
        for (const Student& s : rm_.all()) {
            double pts = 0, worst = 1e9;
            std::string worstCode;
            Risk risk = Risk::LOW;
            for (const SubjectMark& m : s.marks()) {
                double pt = m.internal + model(m.code).predict(m.internal);
                pts += m.credits * GradeScale::fromPercent(pt).points;
                if (pt < worst) { worst = pt; worstCode = m.code; }
                if (classify(pt) > risk) risk = classify(pt);
            }
            os << std::left << std::setw(12) << s.id() << std::setw(18) << s.name().substr(0, 17)
               << std::right << std::setw(8) << pts / s.totalCredits() << std::setw(8) << s.sgpa()
               << std::setw(9) << riskName(risk) << "  " << worstCode << " (pred " << worst << ")\n";
        }
    }
 
    // Forecast for a student whose external exams have not happened yet.
    void forecast(const std::vector<double>& internals, std::ostream& os) const {
        auto subs = ResultManager::catalogue();
        double pts = 0;
        int credits = 0;
        os << std::fixed << std::setprecision(1) << "\n" << std::left << std::setw(9) << "Code" << std::right
           << std::setw(6) << "Int" << std::setw(10) << "PredExt" << std::setw(10) << "PredTot"
           << std::setw(6) << "Gr" << std::setw(9) << "Risk" << "\n" << util::line('-', 50) << "\n";
        for (size_t i = 0; i < subs.size(); ++i) {
            double ext = model(subs[i].code).predict(internals[i]);
            double tot = internals[i] + ext;
            GradeInfo g = GradeScale::fromPercent(tot);
            pts += subs[i].credits * g.points;
            credits += subs[i].credits;
            os << std::left << std::setw(9) << subs[i].code << std::right << std::setw(6) << internals[i]
               << std::setw(10) << ext << std::setw(10) << tot << std::setw(6) << g.letter << std::setw(9)
               << riskName(classify(tot)) << "\n";
        }
        os << std::setprecision(2) << "Predicted SGPA: " << pts / credits << "\n";
    }
};
 
}  // namespace
 
std::string featureName() { return "Performance Predictor & Early Warning"; }
 
void runSpecialFeature(ResultManager& rm) {
    PerformancePredictor pp(rm);        // throws if fewer than 3 students
    while (true) {
        std::cout << "\n--- PERFORMANCE PREDICTOR ---\n 1. Show trained regression models\n"
                  << " 2. Early-warning list (predict from internals)\n 3. Forecast a new student\n 0. Back\n";
        int c = util::readInt("Choice: ", 0, 3);
        if (c == 0) return;
        if (c == 1) pp.printModels(std::cout);
        if (c == 2) pp.printEarlyWarning(std::cout);
        if (c == 3) {
            std::vector<double> in;
            for (const SubjectMark& s : ResultManager::catalogue())
                in.push_back(util::readNumber("Internal marks in " + s.code + " (0-40): ", 0, 40));
            pp.forecast(in, std::cout);
        }
    }
}
```[cite: 2]