import React,{useEffect} from "react";
import { useNavigate } from "react-router-dom";

function Startpage() {
  const navigate = useNavigate();
  useEffect(() => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "/src/components/Startpage.css";
    link.id = "startpage-css";
    document.head.appendChild(link);

    return () => {
      const existingLink = document.getElementById("startpage-css");
      if (existingLink) {
        document.head.removeChild(existingLink);
      }
    };
  }, []);
  return (
    <>
      <div id="webcrumbs">
        <div className="min-h-screen bg-neutral-50">

            {/* Header*/}

          <header className="bg-white shadow-sm">
            <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between h-16 items-center">
                <div className="flex-shrink-0">
                  <span className="text-2xl font-bold bg-gradient-to-r from-primary-600 to-primary-500 bg-clip-text text-transparent">
                    DishaDoot
                  </span>
                </div>
                <div className="hidden md:flex items-center space-x-8">
                  <a
                    href="#"
                    className="hover:text-primary-600 transform hover:-translate-y-0.5 transition-all duration-200"
                  >
                    Home
                  </a>
                  <a
                    href="#features"
                    className="hover:text-primary-600 transform hover:-translate-y-0.5 transition-all duration-200"
                  >
                    Features
                  </a>
                  <a
                    href="#blog"
                    className="hover:text-primary-600 transform hover:-translate-y-0.5 transition-all duration-200"
                  >
                    Blog
                  </a>
                  <a
                    href="#contact"
                    className="hover:text-primary-600 transform hover:-translate-y-0.5 transition-all duration-200"
                  >
                    Contact Us
                  </a>
                  <button className="bg-gradient-to-r from-primary-500 to-primary-600 text-white px-6 py-2 rounded-full hover:shadow-lg transform hover:-translate-y-0.5 transition-all duration-200"   onClick={() => navigate("/login")}>
                    Login
                  </button>
                </div>
              </div>
            </nav>
          </header>

          <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            <section className="text-center mb-20">
              <h1 className="text-5xl font-bold mb-6">
                Welcome to Our Platform
              </h1>
              <p className="text-xl mb-8">
                Discover amazing features and possibilities
              </p>
              <button className="bg-gradient-to-r from-primary-500 to-primary-600 text-white px-8 py-3 rounded-full hover:shadow-lg transform hover:-translate-y-1 transition-all duration-200"onClick={() => navigate("/signup")}>
                Get Started
              </button>
            </section>

            {/* Features*/}

            <section id="features" className="mb-20">
              <h2 className="text-3xl font-bold text-center mb-12">
                Our Features
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                <div className="bg-white p-6 rounded-xl shadow-md hover:shadow-xl transition-all duration-200 transform hover:-translate-y-1">
                  <span className="material-symbols-outlined text-4xl mb-4">
                    rocket_launch
                  </span>
                  <h3 className="text-xl font-semibold mb-3">
                    Fast Performance
                  </h3>
                  <p>
                    Experience lightning-fast performance with our optimized
                    platform.
                  </p>
                </div>
                <div className="bg-white p-6 rounded-xl shadow-md hover:shadow-xl transition-all duration-200 transform hover:-translate-y-1">
                  <span className="material-symbols-outlined text-4xl mb-4">
                    security
                  </span>
                  <h3 className="text-xl font-semibold mb-3">
                    Secure Platform
                  </h3>
                  <p>
                    Your data is protected with enterprise-grade security
                    measures.
                  </p>
                </div>
                <div className="bg-white p-6 rounded-xl shadow-md hover:shadow-xl transition-all duration-200 transform hover:-translate-y-1">
                  <span className="material-symbols-outlined text-4xl mb-4">
                    support_agent
                  </span>
                  <h3 className="text-xl font-semibold mb-3">24/7 Support</h3>
                  <p>Get help anytime with our dedicated support team.</p>
                </div>
              </div>
            </section>

            {/* Blog*/}


            <section id="blog" className="mb-20">
              <h2 className="text-3xl font-bold text-center mb-12">
                Latest Blog Posts
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                <div className="bg-white rounded-xl shadow-md overflow-hidden hover:shadow-xl transition-all duration-200">
                  <img
                    src="https://images.unsplash.com/photo-1498050108023-c5249f4df085"
                    alt="Blog"
                    className="w-full h-48 object-cover"
                  />
                  <div className="p-6">
                    <h3 className="font-semibold text-lg mb-2">
                      Getting Started Guide
                    </h3>
                    <p className="mb-4">
                      Learn how to make the most of our platform with this
                      comprehensive guide.
                    </p>
                    <a
                      href="#"
                      className="text-primary-600 hover:text-primary-700 font-medium"
                    >
                      Read More →
                    </a>
                  </div>
                </div>
                <div className="bg-white rounded-xl shadow-md overflow-hidden hover:shadow-xl transition-all duration-200">
                  <img
                    src="https://images.unsplash.com/photo-1551434678-e076c223a692"
                    alt="Blog"
                    className="w-full h-48 object-cover"
                  />
                  <div className="p-6">
                    <h3 className="font-semibold text-lg mb-2">
                      Platform Updates
                    </h3>
                    <p className="mb-4">
                      Discover the latest features and improvements in our
                      recent update.
                    </p>
                    <a
                      href="#"
                      className="text-primary-600 hover:text-primary-700 font-medium"
                    >
                      Read More →
                    </a>
                  </div>
                </div>
                <div className="bg-white rounded-xl shadow-md overflow-hidden hover:shadow-xl transition-all duration-200">
                  <img
                    src="https://images.unsplash.com/photo-1460925895917-afdab827c52f"
                    alt="Blog"
                    className="w-full h-48 object-cover"
                  />
                  <div className="p-6">
                    <h3 className="font-semibold text-lg mb-2">
                      Success Stories
                    </h3>
                    <p className="mb-4">
                      Read about how our customers achieved success using our
                      platform.
                    </p>
                    <a
                      href="#"
                      className="text-primary-600 hover:text-primary-700 font-medium"
                    >
                      Read More →
                    </a>
                  </div>
                </div>
              </div>
            </section>

            {/* Contact Form*/}
            
            <section id="contact" className="mb-20">
              <h2 className="text-3xl font-bold text-center mb-12">
                Contact Us
              </h2>
              <div className="max-w-2xl mx-auto bg-white rounded-xl shadow-md p-8">
                <form className="space-y-6">
                  <div>
                    <label className="block mb-2 font-medium">Name</label>
                    <input
                      type="text"
                      className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-all duration-200"
                      placeholder="Your name"
                    />
                  </div>
                  <div>
                    <label className="block mb-2 font-medium">Email</label>
                    <input
                      type="email"
                      className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-all duration-200"
                      placeholder="your@email.com"
                    />
                  </div>
                  <div>
                    <label className="block mb-2 font-medium">Message</label>
                    <textarea
                      className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-all duration-200 h-32"
                      placeholder="Your message"
                    ></textarea>
                  </div>
                  <button className="w-full bg-gradient-to-r from-primary-500 to-primary-600 text-white px-6 py-3 rounded-lg hover:shadow-lg transform hover:-translate-y-0.5 transition-all duration-200">
                    Send Message
                  </button>
                </form>
              </div>
            </section>
          </main>

            {/* Footer*/}
          <footer className="bg-white shadow-lg mt-20">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Team</h3>
                  <div className="flex items-center space-x-4">
                    <img
                      src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e"
                      alt="Team Member"
                      className="w-10 h-10 rounded-full transform hover:scale-110 transition-all duration-200"
                    />
                    <div>
                      <p className="font-medium">Royce Dmello</p>
                      <p className="text-sm">CEO</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <img
                      src="https://images.unsplash.com/photo-1438761681033-6461ffad8d80"
                      alt="Team Member"
                      className="w-10 h-10 rounded-full transform hover:scale-110 transition-all duration-200"
                    />
                    <div>
                      <p className="font-medium">Jason Dsozua</p>
                      <p className="text-sm">CTO</p>
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Links</h3>
                  <div className="flex flex-col space-y-2">
                    <a
                      href="#"
                      className="hover:text-primary-600 transform hover:translate-x-1 transition-all duration-200"
                    >
                      About Us
                    </a>
                    <a
                      href="#"
                      className="hover:text-primary-600 transform hover:translate-x-1 transition-all duration-200"
                    >
                      Careers
                    </a>
                    <a
                      href="#"
                      className="hover:text-primary-600 transform hover:translate-x-1 transition-all duration-200"
                    >
                      Support
                    </a>
                  </div>
                </div>
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Contact</h3>
                  <div className="flex items-center space-x-4">
                    <span className="material-symbols-outlined">
                      location_on
                    </span>
                    <p>123 Street, City</p>
                  </div>
                  <div className="flex items-center space-x-4">
                    <span className="material-symbols-outlined">mail</span>
                    <p>contact@example.com</p>
                  </div>
                </div>
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Follow Us</h3>
                  <div className="flex space-x-4">
                    <i className="fa-brands fa-facebook text-2xl hover:text-primary-600 transform hover:scale-110 transition-all duration-200"></i>
                    <i className="fa-brands fa-twitter text-2xl hover:text-primary-400 transform hover:scale-110 transition-all duration-200"></i>
                    <i className="fa-brands fa-instagram text-2xl hover:text-secondary-600 transform hover:scale-110 transition-all duration-200"></i>
                  </div>
                </div>
              </div>
            </div>
          </footer>

        </div>
      </div>
    </>
  );
}

export default Startpage;
